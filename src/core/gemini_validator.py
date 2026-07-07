from __future__ import annotations

import base64
import os
from typing import Optional
import logging

import cv2
import numpy as np

from src.i18n.message_translator import message_translator

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


class GeminiValidator:
    """Dùng Gemini Vision API để fix OCR errors & validate text từ crop image."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(message_translator.get_message("gemini.api_key_missing"))
        self.timeout = timeout
        self._client = None

    def _init_client(self):
        """Lazy init Gemini client (google-genai SDK mới)."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def _generate(self, image_b64: str, prompt: str) -> str:
        """Gọi Gemini API với ảnh + prompt, trả về text."""
        from google.genai import types
        response = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip() if response.text else ""

    def validate_text(
        self,
        crop_image: np.ndarray,
        ocr_text: str,
        label: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        try:
            self._init_client()
            image_b64 = self._encode_image(crop_image)
            prompt = self._build_prompt(ocr_text, label)
            fixed_text = self._generate(image_b64, prompt)
            logger.info(f"Gemini validate_text ({label}): {len(ocr_text)} → {len(fixed_text)} chars")
            return fixed_text if fixed_text else fallback_text
        except Exception as e:
            logger.error(f"Gemini validate_text ({label}) failed: {e}")
            return fallback_text if fallback_text else ocr_text

    def validate_table(self, crop_image: np.ndarray, ocr_text: str) -> str:
        """Specialized validation cho test_table — đọc thẳng từ ảnh."""
        try:
            self._init_client()
            image_b64 = self._encode_image(crop_image)
            prompt = (
                "Bạn là trợ lý OCR bảng xét nghiệm y tế tiếng Việt.\n"
                "Đây là ẢNH bảng kết quả xét nghiệm. OCR tự động đọc sai rất nhiều:\n\n"
                f"OCR thô (tham khảo, không tin cậy):\n{ocr_text}\n\n"
                "YÊU CẦU: Hãy ĐỌC TRỰC TIẾP TỪ ẢNH để trích xuất bảng xét nghiệm.\n"
                "Định dạng output — mỗi chỉ số một dòng:\n"
                "TÊN XÉT NGHIỆM | KẾT QUẢ | ĐƠN VỊ | THAM CHIẾU\n\n"
                "Ví dụ:\n"
                "WBC | 7.2 | K/uL | 4.0-10.0\n"
                "RBC | 4.5 | M/uL | 4.2-5.4\n\n"
                "Lưu ý:\n"
                "- Sửa tên xét nghiệm viết tắt/sai chính tả từ ảnh\n"
                "- Giữ đúng giá trị số từ ảnh\n"
                "- KHÔNG thêm giải thích, chỉ xuất bảng\n"
                "- Nếu ô nào trống thì để trống (vẫn giữ dấu |)"
            )
            fixed_text = self._generate(image_b64, prompt)
            logger.info(f"Gemini validate_table: {len(ocr_text)} → {len(fixed_text)} chars")
            return fixed_text if fixed_text else ocr_text
        except Exception as e:
            logger.error(f"Gemini validate_table failed: {e}")
            return ocr_text

    def validate_batch(self, regions: list[dict]) -> dict[str, str]:
        """
        Xử lý tất cả vùng trong 1 API call duy nhất.
        regions: [{"label": str, "crop_image": ndarray, "ocr_text": str}]
        returns: {label: fixed_text}  — fallback về {} nếu fail
        """
        if not regions:
            return {}
        try:
            self._init_client()
            from google.genai import types
            import json, re as _re

            contents = []
            for r in regions:
                img_b64 = self._encode_image(r["crop_image"])
                contents.append(f"[{r['label']}] OCR thô: {r['ocr_text']}")
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(img_b64), mime_type="image/jpeg"
                ))

            contents.append(self._build_batch_prompt(regions))

            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = response.text.strip() if response.text else ""
            raw = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=_re.MULTILINE).strip()
            result = json.loads(raw)
            valid_labels = {r["label"] for r in regions}
            logger.info(f"Gemini validate_batch: {len(regions)} vùng → 1 API call")
            return {k: str(v) for k, v in result.items() if k in valid_labels}
        except Exception as e:
            logger.error(f"Gemini validate_batch failed: {e}")
            return {}

    @staticmethod
    def _build_batch_prompt(regions: list[dict]) -> str:
        import json
        instructions = {
            "hospital_header": "Chỉ giữ tên BV + địa chỉ + ĐT. Bỏ PID, tên khoa, số bệnh phẩm, mã bệnh án.",
            "patient_info":    "Giữ nhãn trường. Bỏ ký tự rác trước 'Họ tên:'. Mỗi trường 1 dòng.",
            "diagnosis_block": "Giữ nhãn trường. Bỏ ký tự rác đầu. Mỗi trường 1 dòng.",
            "test_table":      "ĐỌC TỪ ẢNH. Mỗi chỉ số 1 dòng: TÊN | KẾT QUẢ | ĐƠN VỊ | THAM CHIẾU.",
            "footer_signature":"Trích chức danh, tên người ký, ngày ký (nếu có).",
        }
        per_region = "\n".join(
            f"- {r['label']}: {instructions.get(r['label'], 'Sửa lỗi OCR, bỏ ký tự rác.')}"
            for r in regions
        )
        keys = json.dumps([r["label"] for r in regions], ensure_ascii=False)
        return (
            f"Bạn là trợ lý OCR y tế. Xử lý {len(regions)} vùng ảnh ở trên:\n"
            f"{per_region}\n\n"
            f"Trả về JSON với key là tên vùng {keys}.\n"
            "Ví dụ: {\"hospital_header\": \"...\", \"patient_info\": \"...\"}\n"
            "CHỈ trả JSON thuần, KHÔNG markdown, KHÔNG giải thích."
        )

    @staticmethod
    def _encode_image(image_bgr: np.ndarray, max_side: int = 800) -> str:
        """Resize về max_side rồi encode JPEG để giảm payload gửi Gemini."""
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            image_bgr = cv2.resize(
                image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.standard_b64encode(buffer).decode("utf-8")

    @staticmethod
    def _build_prompt(ocr_text: str, label: Optional[str]) -> str:
        """Build specialized prompt based on label to clean & fix OCR."""

        if label == "hospital_header":
            return (
                "Bạn là trợ lý OCR tài liệu y tế tiếng Việt.\n"
                "Đây là phần TIÊU ĐỀ BỆNH VIỆN. Hãy đọc ẢNH để xác nhận.\n\n"
                "OCR thô (có thể sai):\n"
                + ocr_text + "\n\n"
                "CHỈ GIỮ LẠI thông tin giới thiệu bệnh viện:\n"
                "- Tên bệnh viện (ví dụ: BỆNH VIỆN TRUNG ƯƠNG HUẾ)\n"
                "- Địa chỉ (số nhà, đường, phường/xã, thành phố)\n"
                "- Số điện thoại (ĐT / Tel)\n\n"
                "BỎ HOÀN TOÀN — KHÔNG được xuất ra:\n"
                "- PID và mọi mã số bệnh nhân\n"
                "- Tên khoa (KHOA XÉT NGHIỆM, KHOA HUYẾT HỌC, KHOA CẤP CỨU, v.v.)\n"
                "- Số bệnh phẩm\n"
                "- Mã bệnh án\n"
                "- Bất kỳ thông tin nào không liên quan đến bệnh viện\n\n"
                "Định dạng output (không thêm nhãn, không giải thích):\n"
                "[Tên bệnh viện]\n"
                "[Địa chỉ]\n"
                "ĐT: [số điện thoại]"
            )

        if label == "patient_info":
            return (
                "Bạn là trợ lý OCR tài liệu y tế tiếng Việt.\n"
                "Đây là phần THÔNG TIN BỆNH NHÂN. Hãy đọc ẢNH để xác nhận.\n\n"
                "OCR thô (có thể có nhiều ký tự rác ở đầu):\n"
                + ocr_text + "\n\n"
                "YÊU CẦU: Trích xuất thông tin bệnh nhân, GIỮ NGUYÊN nhãn trường:\n"
                "Họ tên: [tên đầy đủ]\n"
                "Ngày sinh: [ngày/tháng/năm]\n"
                "Giới tính: [Nam/Nữ]\n"
                "Địa chỉ: [địa chỉ đầy đủ]\n"
                "Số thẻ BHYT: [số thẻ] (nếu có)\n\n"
                "BỎ HOÀN TOÀN:\n"
                "- Tất cả ký tự rác/vô nghĩa xuất hiện TRƯỚC 'Họ tên:'\n"
                "  (ví dụ: 'SỐ TỰ 7 SN Ra NO NGÃ NA ÀAÃ /N 1TYXV', ký hiệu lạ, số ngẫu nhiên)\n"
                "- Ký hiệu đơn lẻ không rõ nghĩa (ị, l, |) đứng lẻ giữa các trường\n\n"
                "Định dạng output: Mỗi trường một dòng. KHÔNG thêm giải thích."
            )

        if label == "diagnosis_block":
            return (
                "Bạn là trợ lý OCR tài liệu y tế tiếng Việt.\n"
                "Đây là phần CHẨN ĐOÁN / THÔNG TIN MẪU. Hãy đọc ẢNH để xác nhận.\n\n"
                "OCR thô (thường có ký tự rác ở đầu):\n"
                + ocr_text + "\n\n"
                "YÊU CẦU: Sửa lỗi OCR và trích xuất, GIỮ NGUYÊN nhãn trường:\n"
                "Chẩn đoán: [...]\n"
                "Khoa/Phòng: [...]\n"
                "Cơ sở yêu cầu: [...]\n"
                "Người lấy mẫu: [...]\n"
                "Thời gian lấy mẫu: [...]\n"
                "Người nhận mẫu: [...]\n"
                "Thời gian nhận mẫu: [...]\n"
                "BS chỉ định: [...]\n"
                "Tình trạng mẫu: [...]\n"
                "Loại bệnh phẩm: [...]\n\n"
                "BỎ HOÀN TOÀN:\n"
                "- Ký tự rác ở đầu văn bản (ví dụ: '3 ĐT TT ky SA SA l', ký hiệu lạ)\n"
                "- Bất kỳ ký hiệu OCR lỗi không thuộc nội dung chẩn đoán\n\n"
                "Định dạng output: Mỗi trường một dòng. Bỏ trường nào không có trong ảnh. KHÔNG thêm giải thích."
            )

        if label == "footer_signature":
            return (
                "Bạn là trợ lý OCR tài liệu y tế tiếng Việt.\n"
                "Đây là phần CHỮ KÝ / KẾT LUẬN cuối phiếu. Hãy đọc ẢNH để xác nhận.\n\n"
                "OCR thô:\n"
                + ocr_text + "\n\n"
                "YÊU CẦU: Sửa lỗi OCR, trích xuất:\n"
                "- Chức danh người ký (Bác sĩ, Trưởng khoa, v.v.)\n"
                "- Họ tên người ký\n"
                "- Ngày ký (nếu có)\n\n"
                "BỎ: Ký tự rác, ký hiệu vô nghĩa, dòng trống thừa.\n"
                "Định dạng output: Text ngắn gọn. KHÔNG thêm giải thích."
            )

        # Default for other labels
        return (
            "Bạn là trợ lý OCR tài liệu y tế tiếng Việt.\n"
            "Sửa lỗi OCR trong đoạn văn bản sau, đọc thêm ẢNH để xác nhận:\n\n"
            + ocr_text + "\n\n"
            "Xóa ký tự rác, ký hiệu vô nghĩa. Chỉ giữ nội dung có nghĩa.\n"
            "Chỉ xuất văn bản đã sửa, KHÔNG giải thích."
        )

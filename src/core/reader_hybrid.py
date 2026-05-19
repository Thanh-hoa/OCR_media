from __future__ import annotations

import os
import re
import threading
import unicodedata
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract

from src.core.gemini_validator import GeminiValidator
from src.utils.image_processing import preprocess_for_ocr

logger = logging.getLogger(__name__)


class HybridReader:
    def __init__(
        self,
        tesseract_cmd: str,
        lang_paddle: str = "vi",
        lang_tesseract: str = "vie",
        confidence_threshold: float = 0.5,
        enable_gemini_validation: bool = True,
    ) -> None:
        self.ocr_paddle = None
        self._paddle_lock = threading.Lock()
        self.lang_paddle = lang_paddle

        exe_path = Path(tesseract_cmd)
        if not exe_path.exists():
            raise FileNotFoundError(f"Tesseract not found: {exe_path}")
        pytesseract.pytesseract.tesseract_cmd = str(exe_path)

        self.lang_tesseract = lang_tesseract
        self.confidence_threshold = confidence_threshold
        self.oem = 3

        # Init Gemini validator (optional, nếu có API key)
        self.gemini_validator = None
        if enable_gemini_validation and os.getenv("GEMINI_API_KEY"):
            try:
                self.gemini_validator = GeminiValidator()
                logger.info("GeminiValidator initialized successfully")
            except ValueError as e:
                logger.warning(f"GeminiValidator init failed: {e}")
        else:
            if not enable_gemini_validation:
                logger.debug("Gemini validation disabled by parameter")
            else:
                logger.debug("GEMINI_API_KEY not set, skipping GeminiValidator")

    def _init_paddle(self) -> None:
        if self.ocr_paddle is None:
            with self._paddle_lock:
                if self.ocr_paddle is None:
                    self.ocr_paddle = PaddleOCR(
                        use_angle_cls=False,
                        lang=self.lang_paddle,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                    )

    def read_ocr_only(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        """Chạy OCR (PaddleOCR + Tesseract fallback) không gọi Gemini."""
        if image_bgr is None or image_bgr.size == 0:
            return ""
        prep = preprocess_for_ocr(image_bgr, label=label)
        text, conf = self._read_paddle(prep, label=label)
        if not text or conf < 0.3:
            fallback = self._read_tesseract(prep, label)
            return fallback if fallback else text
        return text

    def read_text(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        if image_bgr is None or image_bgr.size == 0:
            return ""

        # Optimize: Use ONLY the best preprocessing (preprocess_for_ocr)
        prep_image = preprocess_for_ocr(image_bgr, label=label)

        # Step 1: Try PaddleOCR (primary, faster & more accurate)
        text_paddle, conf_paddle = self._read_paddle(prep_image, label=label)
        logger.debug(f"PaddleOCR ({label}): conf={conf_paddle:.3f}, len={len(text_paddle)}")

        # Step 2: Fallback to Tesseract ONLY if PaddleOCR fails
        if not text_paddle or conf_paddle < 0.3:
            logger.debug(f"PaddleOCR confidence too low ({conf_paddle}), trying Tesseract...")
            text_tesseract = self._read_tesseract(prep_image, label)
            final_text = text_tesseract if text_tesseract else text_paddle
        else:
            final_text = text_paddle

        # Step 3: Validate & fix with Gemini for ALL regions (especially for noise cleanup)
        if self.gemini_validator and final_text:
            logger.debug(f"Calling Gemini validator for {label}...")
            if label == "test_table":
                final_text = self.gemini_validator.validate_table(image_bgr, final_text)
            else:
                # Use validate_text for other regions (with custom prompts)
                final_text = self.gemini_validator.validate_text(
                    image_bgr, final_text, label=label, fallback_text=final_text
                )
            logger.info(f"Gemini fixed {label}: {len(text_paddle)} → {len(final_text)} chars")
        else:
            if not self.gemini_validator:
                logger.warning("Gemini validator not available — trả về OCR thô (kiểm tra GEMINI_API_KEY)")

        return final_text


    def _read_paddle(self, image_bgr: np.ndarray, label: Optional[str] = None) -> tuple[str, float]:
        self._init_paddle()
        try:
            result = self.ocr_paddle.ocr(image_bgr, cls=True)
            
            if not result or not result[0]:
                return "", 0.0
            
            texts = []
            confidences = []
            rows = []
            
            for line in result[0]:
                if not line or len(line) < 2 or not line[1]:
                    continue

                box = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                if confidence <= 0.3 or not text:
                    continue

                avg_y = sum(point[1] for point in box) / 4.0
                avg_x = sum(point[0] for point in box) / 4.0
                rows.append((avg_y, avg_x, text, confidence))

            rows.sort(key=lambda r: (r[0], r[1]))

            for _, _, text, confidence in rows:
                texts.append(text)
                confidences.append(confidence)
            
            if not texts:
                return "", 0.0
            
            if label == "test_table":
                full_text = self._format_table_rows(rows, image_bgr.shape[1])
            elif label == "hospital_header":
                # Chỉ lấy cột trái (tên BV, địa chỉ, ĐT) — bỏ cột phải (PID, KHOA, Mã bệnh án)
                mid_x = image_bgr.shape[1] * 0.55
                left_rows = [(y, x, t, c) for y, x, t, c in rows if x <= mid_x]
                full_text = " ".join(t for _, _, t, _ in sorted(left_rows, key=lambda r: r[0]))
            elif label == "diagnosis_block":
                full_text = self._format_two_column(rows, image_bgr.shape[1])
            else:
                full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return self._normalize_text(full_text, label=label), avg_confidence
            
        except Exception as e:
            return "", 0.0

    def _read_tesseract(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        try:
            config = self._build_config(label)
            text = pytesseract.image_to_string(image_bgr, lang=self.lang_tesseract, config=config)
            return self._normalize_text(text)
        except Exception as e:
            return ""

    def _build_config(self, label: Optional[str]) -> str:
        psm_map = {
            "hospital_header": 11,  # ← Changed from 5 to 11 (sparse text with layout)
            "patient_info": 6,
            "diagnosis_block": 6,
            "test_table": 6,
            "footer_signature": 7,
        }
        psm = psm_map.get(label, 6)
        return f"--oem {self.oem} --psm {psm} -c preserve_interword_spaces=1"



    @staticmethod
    def _format_two_column(rows: list[tuple[float, float, str, float]], image_width: int) -> str:
        """Tách text của vùng có 2 cột (trái / phải) thay vì trộn lẫn theo Y.

        Ví dụ hospital_header:
          Cột trái  — tên bệnh viện, địa chỉ, SĐT
          Cột phải  — PID, số bệnh phẩm, mã bệnh án

        Ngưỡng phân cột: 55% chiều rộng ảnh crop để ưu tiên cột trái rộng hơn.
        Output: "<left_text> | <right_text>" hoặc chỉ một cột nếu cột kia trống.
        """
        mid_x = image_width * 0.55
        left_rows  = [(y, x, t) for y, x, t, _ in rows if x <= mid_x]
        right_rows = [(y, x, t) for y, x, t, _ in rows if x > mid_x]

        left_text  = " ".join(t for _, _, t in sorted(left_rows,  key=lambda r: r[0]))
        right_text = " ".join(t for _, _, t in sorted(right_rows, key=lambda r: r[0]))

        parts = [p.strip() for p in (left_text, right_text) if p.strip()]
        return " | ".join(parts)

    @staticmethod
    def _format_table_rows(rows: list[tuple[float, float, str, float]], image_width: int) -> str:
        if not rows:
            return ""

        row_threshold = 24.0
        grouped: list[list[tuple[float, float, str, float]]] = []

        for item in rows:
            if not grouped:
                grouped.append([item])
                continue

            prev_y = sum(r[0] for r in grouped[-1]) / len(grouped[-1])
            if abs(item[0] - prev_y) <= row_threshold:
                grouped[-1].append(item)
            else:
                grouped.append([item])

        lines: list[str] = []
        gap_threshold = max(80.0, image_width * 0.08)

        for row_items in grouped:
            row_sorted = sorted(row_items, key=lambda r: r[1])
            row_parts: list[str] = []
            prev_x: Optional[float] = None
            for _, x, text, _ in row_sorted:
                token = text.strip()
                if not token:
                    continue
                if prev_x is not None and (x - prev_x) > gap_threshold and row_parts:
                    row_parts.append("|")
                row_parts.append(token)
                prev_x = x

            row_text = " ".join(row_parts).strip()
            if row_text:
                lines.append(row_text)

        return "\n".join(lines)

    @staticmethod
    def _normalize_text(text: str, label: Optional[str] = None) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # C\u1eaft b\u1ecf k\u00fd t\u1ef1 r\u00e1c tr\u01b0\u1edbc nh\u00e3n quan tr\u1ecdng (fallback khi Gemini kh\u00f4ng ch\u1ea1y)
        if label == "patient_info":
            m = re.search(r"H\u1ecd\s*t\u00ean\s*:", text)
            if m:
                text = text[m.start():]
        elif label == "diagnosis_block":
            m = re.search(r"Ch\u1ea9n\s*\u0111o\u00e1n\s*:", text)
            if m:
                text = text[m.start():]

        if label == "test_table":
            text = text.replace("\r", "\n")
            text = re.sub(r"\n+", "\n", text)
            lines = []
            for line in text.split("\n"):
                line = re.sub(r"\s+", " ", line).strip()
                line = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\-/:.,()%| ]", "", line)
                line = re.sub(r"([:.,%/\-|])\1+", r"\1", line)
                if line:
                    lines.append(line)
            return "\n".join(lines)

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        cleaned_tokens = []
        for token in text.split(" "):
            token = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\-/:.,()%]", "", token)
            if not token:
                continue

            letter_count = sum(ch.isalpha() for ch in token)
            digit_count = sum(ch.isdigit() for ch in token)
            punct_count = sum(ch in "-/:.,()%" for ch in token)
            if letter_count + digit_count == 0:
                continue
            if punct_count > max(letter_count + digit_count, 1):
                continue

            cleaned_tokens.append(token)

        text = " ".join(cleaned_tokens).strip()
        text = re.sub(r"([:.,%/\-])\1+", r"\1", text)
        return text


MedicalReaderHybrid = HybridReader

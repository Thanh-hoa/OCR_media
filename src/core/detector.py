from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionItem:
    label: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    cropped_image: np.ndarray


class MedicalDetector:
    CLASS_NAMES = {
        0: "diagnosis_block",
        1: "footer_signature",
        2: "hospital_header",
        3: "patient_info",
        4: "test_table",
    }
    
    def __init__(
        self, model_path: str, conf_threshold: float = 0.25, crop_padding_ratio: float = 0.02
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_file}")

        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.crop_padding_ratio = crop_padding_ratio

    def detect(self, image_bgr: np.ndarray) -> List[DetectionItem]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ.")

        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)

        if not results:
            return []

        result = results[0]

        use_obb = result.obb is not None
        if use_obb:
            boxes_obj = result.obb
            # xyxyxyxy: tensor (N, 4, 2) — 4 góc thực của từng OBB
            obb_corners = result.obb.xyxyxyxy
        elif result.boxes is not None:
            boxes_obj = result.boxes
            obb_corners = None
        else:
            return []

        detections: List[DetectionItem] = []

        for i, box in enumerate(boxes_obj):
            # bbox_xyxy dùng để trả về metadata trong JSON, không dùng để crop
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_i, y1_i, x2_i, y2_i = self._clamp_box(
                int(x1), int(y1), int(x2), int(y2), image_bgr.shape
            )

            if use_obb and obb_corners is not None:
                # Crop theo 4 góc thực của OBB — không bị phình khi ảnh nghiêng
                pts = obb_corners[i].cpu().numpy()  # (4, 2)
                cropped_image = self._crop_obb_region(image_bgr, pts)
            else:
                cropped_image = image_bgr[y1_i:y2_i, x1_i:x2_i]

            if cropped_image is None or cropped_image.size == 0:
                continue

            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = self.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            detections.append(
                DetectionItem(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1_i, y1_i, x2_i, y2_i),
                    cropped_image=cropped_image,
                )
            )

        return detections

    @staticmethod
    def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Không thể giải mã ảnh. Hãy kiểm tra tệp tải lên.")
        return image_bgr

    @staticmethod
    def _crop_obb_region(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Crop vùng OBB bằng perspective transform.

        Thay vì lấy axis-aligned XYXY (bị phình to khi doc nghiêng),
        dùng 4 góc thực của OBB để warpPerspective ra ảnh chữ nhật thẳng.

        pts: shape (4, 2) — 4 góc của OBB theo thứ tự bất kỳ từ YOLO.
        """
        h_img, w_img = image_bgr.shape[:2]
        pts = pts.astype(np.float32)

        # Clamp vào biên ảnh phòng YOLO trả tọa độ vượt ngoài
        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

        # Sắp xếp 4 góc: TL (sum nhỏ nhất), BR (sum lớn nhất),
        #                  TR (diff nhỏ nhất), BL (diff lớn nhất)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        ordered = np.array([tl, tr, br, bl], dtype=np.float32)

        w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        h = int(max(np.linalg.norm(br - tr), np.linalg.norm(bl - tl)))

        if w <= 0 or h <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

    @staticmethod
    def _clamp_box(
        x1: int, y1: int, x2: int, y2: int, image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))

        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _add_padding(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_shape: Tuple[int, int, int],
        padding_ratio: float,
    ) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)
        pad_x = max(pad_x, 4)
        pad_y = max(pad_y, 4)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return x1, y1, x2, y2


YoloDetector = MedicalDetector

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        crop_padding_ratio: float = 0.02,
        imgsz: int = 640,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_file}")

        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray) -> List[DetectionItem]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ.")

        results = self.model.predict(
            image_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

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
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = self.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

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

            refined_crop = self._refine_crop(cropped_image, label)
            if refined_crop is not None and refined_crop.size > 0:
                cropped_image = refined_crop

            detections.append(
                DetectionItem(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1_i, y1_i, x2_i, y2_i),
                    cropped_image=cropped_image,
                )
            )

        return self._dedupe_detections(detections)

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

    @classmethod
    def _dedupe_detections(cls, detections: List[DetectionItem]) -> List[DetectionItem]:
        """Keep the best logical region for each configured class."""
        best_by_label: Dict[str, DetectionItem] = {}
        unknown: List[DetectionItem] = []

        for item in detections:
            if item.label.startswith("unknown_"):
                unknown.append(item)
                continue

            current = best_by_label.get(item.label)
            if current is None:
                best_by_label[item.label] = item
                continue

            if item.label == "test_table":
                item_area = item.cropped_image.shape[0] * item.cropped_image.shape[1]
                current_area = current.cropped_image.shape[0] * current.cropped_image.shape[1]
                if item_area > current_area:
                    best_by_label[item.label] = item
            elif item.confidence > current.confidence:
                best_by_label[item.label] = item

        return list(best_by_label.values()) + unknown

    @classmethod
    def _refine_crop(cls, image_bgr: np.ndarray, label: str) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr
        if label == "test_table":
            return cls._refine_table_crop(image_bgr)
        if label in {
            "hospital_header",
            "patient_info",
            "diagnosis_block",
            "footer_signature",
        }:
            return cls._refine_text_block_crop(image_bgr)
        return image_bgr

    @staticmethod
    def _refine_table_crop(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )

        h, w = binary_inv.shape
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(30, w // 22), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(20, h // 22))
        )
        horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel)
        grid_mask = cv2.bitwise_or(horizontal, vertical)
        grid_mask = cv2.dilate(
            grid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int, float]] = []
        min_area = w * h * 0.08
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area:
                continue
            if bw < w * 0.35 or bh < h * 0.25:
                continue
            candidates.append((x, y, bw, bh, float(area)))

        if not candidates:
            return image_bgr

        x, y, bw, bh, _ = max(candidates, key=lambda item: item[4])
        pad_x = max(8, int(bw * 0.015))
        pad_y = max(6, int(bh * 0.015))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        return image_bgr[y1:y2, x1:x2]

    @staticmethod
    def _refine_text_block_crop(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        h, w = binary_inv.shape
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(12, w // 45), max(2, h // 180))
        )
        connected = cv2.dilate(binary_inv, kernel, iterations=1)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        min_area = max(20, int(w * h * 0.0002))
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < min_area:
                continue
            if bw < 3 or bh < 3:
                continue
            boxes.append((x, y, x + bw, y + bh))

        if not boxes:
            return image_bgr

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        pad_x = max(8, int((x2 - x1) * 0.04))
        pad_y = max(6, int((y2 - y1) * 0.12))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        refined = image_bgr[y1:y2, x1:x2]
        if refined.shape[0] < 12 or refined.shape[1] < 12:
            return image_bgr
        return refined

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

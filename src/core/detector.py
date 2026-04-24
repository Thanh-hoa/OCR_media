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
    # Class name mapping for numbered labels (0-4)
    # MUST match model.names order from training!
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
            raise FileNotFoundError(f"Khong tim thay model tai: {model_file}")

        print(f"\n[DEBUG] Loading model from: {model_file}")
        print(f"[DEBUG] Model file size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        self.model = YOLO(str(model_file))
        print(f"[DEBUG] Model loaded successfully!")
        print(f"[DEBUG] Model task: {self.model.task}")
        print(f"[DEBUG] Model classes: {self.model.names}")
        print(f"[DEBUG] Using CLASS_NAMES mapping: {self.CLASS_NAMES}\n")
        
        self.conf_threshold = conf_threshold
        self.crop_padding_ratio = crop_padding_ratio

    def detect(self, image_bgr: np.ndarray) -> List[DetectionItem]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Anh dau vao khong hop le.")

        print(f"\n[DEBUG] Running detection on image shape: {image_bgr.shape}")
        print(f"[DEBUG] Confidence threshold: {self.conf_threshold}")
        
        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)
        
        print(f"[DEBUG] Prediction results count: {len(results)}")
        
        if not results:
            print("[DEBUG] No results returned!")
            return []

        result = results[0]
        print(f"[DEBUG] Result has OBB: {result.obb is not None}")
        print(f"[DEBUG] Result has boxes: {result.boxes is not None}")
        
        # Determine which bounding box format to use
        if result.obb is not None:
            # OBB format (Oriented Bounding Boxes)
            boxes_obj = result.obb
            print(f"[DEBUG] Using OBB format - boxes count: {len(boxes_obj)}")
        elif result.boxes is not None:
            # Regular format
            boxes_obj = result.boxes
            print(f"[DEBUG] Using regular XYXY format - boxes count: {len(boxes_obj)}")
        else:
            print("[DEBUG] No boxes or OBB detected!")
            return []

        names = result.names or {}
        detections: List[DetectionItem] = []

        for i, box in enumerate(boxes_obj):
            # Extract coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Test mode: keep raw YOLO bbox (no extra padding).
            # Uncomment this block to re-enable padding after evaluating crop quality.
            # x1, y1, x2, y2 = self._add_padding(
            #     int(x1), int(y1), int(x2), int(y2), image_bgr.shape, self.crop_padding_ratio
            # )
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1_i, y1_i, x2_i, y2_i = self._clamp_box(
                x1, y1, x2, y2, image_bgr.shape
            )

            cropped_image = image_bgr[y1_i:y2_i, x1_i:x2_i]
            if cropped_image.size == 0:
                print(f"[DEBUG] Box {i}: Cropped image is empty, skipping")
                continue

            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            
            # Map class ID to region name
            label = self.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            print(f"[DEBUG] Box {i}: label={label} (cls_id={cls_id}), conf={conf:.3f}, bbox=({x1_i},{y1_i},{x2_i},{y2_i})")

            detections.append(
                DetectionItem(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(x1_i, y1_i, x2_i, y2_i),
                    cropped_image=cropped_image,
                )
            )

        print(f"[DEBUG] Total detections: {len(detections)}\n")
        return detections

    @staticmethod
    def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Khong the decode anh. Hay kiem tra file upload.")
        return image_bgr

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
        # Keep some minimum padding to avoid clipping Vietnamese accents.
        pad_x = max(pad_x, 4)
        pad_y = max(pad_y, 4)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return x1, y1, x2, y2


# Backward compatible alias for old imports.
YoloDetector = MedicalDetector

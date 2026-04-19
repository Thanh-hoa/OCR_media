from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

import cv2

from src.core.detector import DetectionItem


def _safe_filename_stem(name: str, max_len: int = 64) -> str:
    stem = Path(name).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:max_len] if len(safe) > max_len else safe or "image"


def _color_for_label(label: str) -> tuple[int, int, int]:
    h = hash(label) & 0xFFFFFF
    b = (h & 0xFF)
    g = (h >> 8) & 0xFF
    r = (h >> 16) & 0xFF
    return int(b), int(g), int(r)


def save_yolo_region_debug(
    image_bgr: np.ndarray,
    detections: List[DetectionItem],
    original_filename: str,
    output_dir: Path,
    project_root: Path,
) -> dict[str, str]:
    """
    Luu anh goc co ve bbox + tung crop de kiem tra YOLO phan vung.
    Tra ve duong dan file (relative) de API co the tra ve neu can.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename_stem(original_filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{ts}_{stem}"

    paths: dict[str, str] = {"annotated": "", "crops_dir": ""}

    vis = image_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        color = _color_for_label(item.label)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        caption = f"{item.label} {item.confidence:.2f}"
        cv2.putText(
            vis,
            caption,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    annotated_path = output_dir / f"{prefix}_annotated.jpg"
    cv2.imwrite(str(annotated_path), vis)
    paths["annotated"] = str(annotated_path.relative_to(project_root)).replace("\\", "/")

    crops_root = output_dir / "crops" / prefix
    crops_root.mkdir(parents=True, exist_ok=True)
    paths["crops_dir"] = str(crops_root.relative_to(project_root)).replace("\\", "/")

    for idx, item in enumerate(detections):
        crop_name = f"{idx:02d}_{item.label}_{item.confidence:.3f}.jpg"
        crop_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in crop_name)
        cv2.imwrite(str(crops_root / crop_name), item.cropped_image)

    return paths


def should_save_yolo_debug() -> bool:
    return os.getenv("YOLO_DEBUG_SAVE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

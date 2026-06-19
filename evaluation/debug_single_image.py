"""
Debug YOLO crop + OCR for one image.

Usage:
    python evaluation/debug_single_image.py path/to/image.jpg

Outputs:
    evaluation/debug_single/<image_stem>/annotated.jpg
    evaluation/debug_single/<image_stem>/<label>.jpg
    evaluation/debug_single/<image_stem>/ocr_result.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.core.detector import MedicalDetector
from src.core.reader_hybrid import HybridReader
from src.utils.image_processing import deskew_image


YOLO_MODEL = ROOT / "models" / "weights" / "best.pt"
TESSERACT_EXE = Path(r"D:\HK2_4\DoAn\OCR\tesseract.exe")


COLORS = {
    "hospital_header": (255, 0, 0),
    "patient_info": (0, 255, 255),
    "diagnosis_block": (0, 0, 255),
    "test_table": (255, 255, 0),
    "footer_signature": (0, 128, 255),
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluation/debug_single_image.py path/to/image.jpg")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_h, original_w = image.shape[:2]
    image = deskew_image(image)

    detector = MedicalDetector(model_path=str(YOLO_MODEL), conf_threshold=0.25)
    reader = HybridReader(
        tesseract_cmd=str(TESSERACT_EXE),
        enable_gemini_validation=False,
    )

    detections = detector.detect(image)

    out_dir = ROOT / "evaluation" / "debug_single" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    results = {
        "image": image_path.name,
        "original_size": {"width": original_w, "height": original_h},
        "deskewed_size": {"width": image.shape[1], "height": image.shape[0]},
        "detections": [],
    }

    # Keep the highest-confidence crop per label for OCR output.
    best_by_label = {}
    for det in detections:
        if det.label not in best_by_label or det.confidence > best_by_label[det.label].confidence:
            best_by_label[det.label] = det

        x1, y1, x2, y2 = det.bbox_xyxy
        color = COLORS.get(det.label, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)

    for label, det in sorted(best_by_label.items()):
        crop_path = out_dir / f"{label}.jpg"
        cv2.imwrite(str(crop_path), det.cropped_image)
        text = reader.read_ocr_only(det.cropped_image, label)
        h, w = det.cropped_image.shape[:2]
        results["detections"].append(
            {
                "label": label,
                "confidence": round(det.confidence, 6),
                "bbox_xyxy": list(det.bbox_xyxy),
                "crop_size": {"width": w, "height": h},
                "crop_file": str(crop_path.relative_to(ROOT)),
                "text": text,
            }
        )

    result_path = out_dir / "ocr_result.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input size: {original_w}x{original_h}")
    print(f"Detections: {len(detections)}")
    print(f"Output dir: {out_dir}")
    print(f"Result JSON: {result_path}")
    for item in results["detections"]:
        print()
        print(f"[{item['label']}] conf={item['confidence']} crop={item['crop_size']}")
        print(item["text"])


if __name__ == "__main__":
    main()

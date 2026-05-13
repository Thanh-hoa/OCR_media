from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.core.detector import MedicalDetector
from src.core.reader_hybrid import HybridReader
from src.utils.image_processing import deskew_image
from src.utils.yolo_debug import save_yolo_region_debug, should_save_yolo_debug

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables từ .env
project_root = Path(__file__).resolve().parents[2]
env_file = project_root / ".env"
logger.info(f"Loading .env from: {env_file} (exists: {env_file.exists()})")
load_dotenv(dotenv_path=str(env_file))

app = FastAPI(
    title="Project OCR API",
    description="Upload image -> YOLOv11 detection -> Hybrid OCR",
    version="2.0.0",
)

PROJECT_ROOT = project_root
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "best.pt"
TESSERACT_EXE = Path(r"D:\HK2_4\DoAn\OCR\tesseract.exe")
YOLO_REGIONS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_regions"


detector = MedicalDetector(model_path=str(YOLO_MODEL_PATH), conf_threshold=0.25)
reader = HybridReader(tesseract_cmd=str(TESSERACT_EXE), confidence_threshold=0.5)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_CLASS_ORDER = [
    "hospital_header",
    "patient_info",
    "diagnosis_block",
    "test_table",
    "footer_signature",
]

@app.get("/health")
def health() -> Dict[str, Any]:
    gemini_enabled = bool(os.getenv("GEMINI_API_KEY"))
    gemini_initialized = reader.gemini_validator is not None
    return {
        "status": "ok",
        "model_path": str(YOLO_MODEL_PATH),
        "tesseract_path": str(TESSERACT_EXE),
        "yolo_debug_dir": str(YOLO_REGIONS_OUTPUT_DIR),
        "gemini_api_key_set": gemini_enabled,
        "gemini_validator_initialized": gemini_initialized,
        "how_to_test_postman": {
            "method": "POST",
            "url": "/v1/ocr/upload",
            "body": "form-data",
            "key": "file (type: File)",
        },
    }

@app.post("/v1/ocr/upload")
async def upload_and_ocr(file: UploadFile = File(...)) -> Dict[str, Any]:
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Định dạng file không được hỗ trợ. "
                "Hãy dùng: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp"
            ),
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise ValueError("File rỗng. Hãy chọn một ảnh hợp lệ.")
        image_bgr = MedicalDetector.decode_image_bytes(file_bytes)
        image_bgr = deskew_image(image_bgr)
        logger.info(f"Processing file: {filename}")
        detections = detector.detect(image_bgr)
        logger.info(f"Detected {len(detections)} regions")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Image processing error: {exc}")
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh: {exc}") from exc

    debug_output: Optional[Dict[str, str]] = None
    if should_save_yolo_debug() and detections:
        debug_output = save_yolo_region_debug(
            image_bgr,
            detections,
            filename,
            YOLO_REGIONS_OUTPUT_DIR,
            PROJECT_ROOT,
        )

    results: List[Dict[str, Any]] = []
    for item in detections:
        logger.info(f"Reading text for {item.label}...")
        text = reader.read_text(item.cropped_image, label=item.label)
        logger.info(f"Result for {item.label}: {len(text)} chars")
        results.append(
            {
                "label": item.label,
                "confidence": round(item.confidence, 6),
                "bbox_xyxy": list(item.bbox_xyxy),
                "text": text,
            }
        )

    result = _group_regions(results)
    payload: Dict[str, Any] = {
        "filename": filename,
        "result": result,
    }
    if debug_output:
        payload["debug_output"] = debug_output
    logger.info(f"OCR processing complete for {filename}")
    return payload

def _group_regions(regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for region in regions:
        class_name = str(region.get("label", "")).strip()
        if class_name:
            groups[class_name].append(region)

    output: Dict[str, Any] = {}
    for class_name in OUTPUT_CLASS_ORDER:
        items = groups.get(class_name, [])
        items_sorted = sorted(
            items,
            key=lambda x: (
                x.get("bbox_xyxy", [0, 0, 0, 0])[1],
                x.get("bbox_xyxy", [0, 0, 0, 0])[0],
            ),
        )

        merged_text = " ".join(
            item.get("text", "").strip() for item in items_sorted if item.get("text", "").strip()
        ).strip()
        avg_conf = (
            round(sum(float(item.get("confidence", 0.0)) for item in items_sorted) / len(items_sorted), 6)
            if items_sorted
            else 0.0
        )

        output[class_name] = {
            "count": len(items_sorted),
            "confidence_avg": avg_conf,
            "text": merged_text,
            "boxes": [item.get("bbox_xyxy", []) for item in items_sorted],
        }

    return output

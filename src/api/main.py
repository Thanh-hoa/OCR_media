from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.data_parser import MedicalRecordParser
from src.core.detector import MedicalDetector
from src.core.exceptions import AppException
from src.core.reader_hybrid import HybridReader
from src.i18n.message_translator import message_translator
from src.utils.image_processing import deskew_image


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

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://medicalocr-nthoa.io.vn,https://api.medicalocr-nthoa.io.vn",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
    locale = message_translator.resolve_locale(request.headers.get("accept-language"))
    message = message_translator.get_message(exc.message_key, locale=locale, **exc.params)
    return JSONResponse(status_code=exc.status_code, content={"isError": True, "message": message})

PROJECT_ROOT = project_root
YOLO_MODEL_PATH = Path(
    os.getenv("YOLO_MODEL_PATH", str(PROJECT_ROOT / "models" / "weights" / "best.pt"))
)
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_EXE") or "tesseract"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


detector = MedicalDetector(model_path=str(YOLO_MODEL_PATH), conf_threshold=0.25)
reader = HybridReader(tesseract_cmd=TESSERACT_CMD, confidence_threshold=0.5)
ocr_parser = MedicalRecordParser()

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
    tesseract_resolved = shutil.which(TESSERACT_CMD) or TESSERACT_CMD
    tesseract_exists = bool(shutil.which(TESSERACT_CMD)) or Path(TESSERACT_CMD).exists()
    return {
        "status": "ok",
        "model_path": str(YOLO_MODEL_PATH),
        "model_exists": YOLO_MODEL_PATH.exists(),
        "tesseract_cmd": TESSERACT_CMD,
        "tesseract_resolved": tesseract_resolved,
        "tesseract_available": tesseract_exists,
        "max_upload_mb": MAX_UPLOAD_MB,
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
        raise AppException(
            "upload.unsupported_format",
            params={"allowed": ", ".join(sorted(ALLOWED_EXTENSIONS))},
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise AppException("upload.empty_file")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise AppException("upload.file_too_large", params={"max_mb": MAX_UPLOAD_MB})
        image_bgr = MedicalDetector.decode_image_bytes(file_bytes)
        image_bgr = deskew_image(image_bgr)
        logger.info(f"Processing file: {filename}")
        detections = detector.detect(image_bgr)
        logger.info(f"Detected {len(detections)} regions")
    except AppException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Image processing error: {exc}")
        raise AppException("upload.processing_error", params={"error": str(exc)}) from exc

    # Bước 1: OCR song song (không Gemini) — nhanh
    async def _ocr_one(item):
        text = await asyncio.to_thread(reader.read_ocr_only, item.cropped_image, item.label)
        return {"label": item.label, "confidence": round(item.confidence, 6),
                "bbox_xyxy": list(item.bbox_xyxy), "text": text,
                "_crop": item.cropped_image}

    ocr_results: List[Dict[str, Any]] = list(
        await asyncio.gather(*[_ocr_one(item) for item in detections])
    )

    # Bước 2: 1 Gemini call duy nhất cho tất cả vùng
    if reader.gemini_validator:
        batch_input = [
            {"label": r["label"], "crop_image": r["_crop"], "ocr_text": r["text"]}
            for r in ocr_results
        ]
        gemini_out = await asyncio.to_thread(
            reader.gemini_validator.validate_batch, batch_input
        )
        for r in ocr_results:
            if r["label"] in gemini_out:
                r["text"] = gemini_out[r["label"]]
        logger.info(f"Gemini batch done: {list(gemini_out.keys())}")

    results: List[Dict[str, Any]] = [
        {k: v for k, v in r.items() if k != "_crop"} for r in ocr_results
    ]

    result = _group_regions(results)
    logger.info(f"OCR processing complete for {filename}")

    ocr_texts = {k: v.get("text") for k, v in result.items()}
    try:
        parsed = ocr_parser.parse(ocr_data=ocr_texts)
    except Exception as exc:
        logger.warning(f"Structured parse failed: {exc}")
        parsed = None

    return {"filename": filename, "result": result, "parsedData": parsed}

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

        merged_text = "\n".join(
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

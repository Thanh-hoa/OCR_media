"""
Đánh giá 4 cấu hình OCR bằng CER, WER, Field Accuracy.

Cách chạy:
    python evaluation/evaluate.py

Yêu cầu:
    pip install editdistance
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
from dotenv import load_dotenv

try:
    import editdistance
except ModuleNotFoundError:
    editdistance = None

# --- Path setup ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.core.data_parser import MedicalRecordParser
from src.core.detector import MedicalDetector
from src.core.reader import MedicalReader
from src.core.reader_hybrid import HybridReader
from src.utils.image_processing import deskew_image

# --- Config ---
YOLO_MODEL    = ROOT / "models" / "weights" / "best.pt"
TESSERACT_EXE = Path(r"D:\HK2_4\DoAn\OCR\tesseract.exe")
GT_FILE       = Path(__file__).parent / "ground_truth.json"
IMG_DIR       = Path(__file__).parent / "images"

REGIONS = ["hospital_header", "patient_info", "diagnosis_block", "test_table", "footer_signature"]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_cer(pred: str, ref: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(ref_chars)."""
    pred = pred.strip()
    ref  = ref.strip()
    if not ref:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, ref) / len(ref)


def compute_wer(pred: str, ref: str) -> float:
    """Word Error Rate = edit_distance(words) / len(ref_words)."""
    pred_words = pred.strip().split()
    ref_words  = ref.strip().split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return _edit_distance(pred_words, ref_words) / len(ref_words)


def _edit_distance(a, b) -> int:
    if editdistance is not None:
        return editdistance.eval(a, b)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, a_item in enumerate(a, start=1):
        current = [i]
        for j, b_item in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (a_item != b_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def field_match(pred_val: Optional[str], gt_val: Optional[str]) -> bool:
    """So sánh field (chuẩn hóa lowercase, bỏ khoảng trắng thừa)."""
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    return pred_val.strip().lower() == gt_val.strip().lower()


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def crop_regions(detector: MedicalDetector, img_path: Path) -> Dict[str, object]:
    """Detect & crop vùng từ ảnh. Trả về dict[label -> cropped_bgr]."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img = deskew_image(img)
    detections = detector.detect(img)
    return {d.label: d.cropped_image for d in detections}


def run_config(
    config_name: str,
    crops: Dict[str, object],
    reader_fn,
    parser: MedicalRecordParser,
    gt_item: dict,
) -> dict:
    """
    Chạy 1 cấu hình OCR trên crops, tính CER/WER/Field Accuracy.
    reader_fn(crop, label) -> str
    """
    gt_regions = gt_item.get("regions", {})
    gt_fields  = gt_item.get("fields", {})

    region_cer, region_wer = [], []
    ocr_texts: Dict[str, str] = {}

    for label in REGIONS:
        crop = crops.get(label)
        gt_text = gt_regions.get(label, "")

        if crop is None:
            # YOLO không detect được vùng này
            pred_text = ""
        else:
            t0 = time.time()
            pred_text = reader_fn(crop, label)
            elapsed = time.time() - t0

        ocr_texts[label] = pred_text

        if gt_text:
            region_cer.append(compute_cer(pred_text, gt_text))
            region_wer.append(compute_wer(pred_text, gt_text))

    # Field Accuracy: parse OCR text → structured fields
    parsed = parser.parse(ocr_texts)
    extracted = parsed.get("extractedData", {})

    correct, total = 0, 0
    field_details = {}
    for field, gt_val in gt_fields.items():
        pred_val = extracted.get(field)
        matched = field_match(pred_val, gt_val)
        field_details[field] = {
            "gt": gt_val,
            "pred": pred_val,
            "match": matched,
        }
        correct += int(matched)
        total += 1

    return {
        "config": config_name,
        "CER": sum(region_cer) / len(region_cer) if region_cer else 0.0,
        "WER": sum(region_wer) / len(region_wer) if region_wer else 0.0,
        "field_accuracy": correct / total if total else 0.0,
        "field_details": field_details,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Khởi tạo YOLO detector...")
    detector = MedicalDetector(model_path=str(YOLO_MODEL), conf_threshold=0.25)
    parser   = MedicalRecordParser()

    print("Khởi tạo readers...")
    # Cấu hình 1: Tesseract only
    tess_reader = MedicalReader(tesseract_cmd=str(TESSERACT_EXE))

    # Cấu hình 2, 3: PaddleOCR only / PaddleOCR + fallback (không Gemini)
    hybrid_no_gemini = HybridReader(
        tesseract_cmd=str(TESSERACT_EXE),
        enable_gemini_validation=False,
    )

    # Cấu hình 4: PaddleOCR + fallback + Gemini
    hybrid_gemini = HybridReader(
        tesseract_cmd=str(TESSERACT_EXE),
        enable_gemini_validation=True,
    )

    CONFIGS = [
        ("Tesseract only",              lambda img, lbl: tess_reader.read_text(img, lbl)),
        ("PaddleOCR only",              lambda img, lbl: hybrid_no_gemini.read_paddle_only(img, lbl)),
        ("PaddleOCR + Tesseract",       lambda img, lbl: hybrid_no_gemini.read_ocr_only(img, lbl)),
        ("PaddleOCR + Tesseract + Gemini", lambda img, lbl: hybrid_gemini.read_text(img, lbl)),
    ]

    with open(GT_FILE, encoding="utf-8") as f:
        ground_truth = json.load(f)

    all_results: list[dict] = []

    for gt_item in ground_truth:
        img_path = IMG_DIR / gt_item["image"]
        print(f"\n=== Ảnh: {gt_item['image']} ===")

        try:
            crops = crop_regions(detector, img_path)
        except FileNotFoundError as e:
            print(f"  [BỎ QUA] {e}")
            continue

        print(f"  Detect được {len(crops)} vùng: {list(crops.keys())}")

        for config_name, reader_fn in CONFIGS:
            result = run_config(config_name, crops, reader_fn, parser, gt_item)
            result["image"] = gt_item["image"]
            all_results.append(result)
            print(
                f"  [{config_name}] "
                f"CER={result['CER']:.1%}  WER={result['WER']:.1%}  "
                f"FieldAcc={result['field_accuracy']:.1%}"
            )

    # ── Tổng hợp kết quả ──
    print("\n" + "=" * 70)
    print(f"{'CẤU HÌNH':<35} {'CER':>8} {'WER':>8} {'FIELD ACC':>10}")
    print("-" * 70)

    from collections import defaultdict
    agg: dict[str, list] = defaultdict(lambda: {"CER": [], "WER": [], "FA": []})

    for r in all_results:
        agg[r["config"]]["CER"].append(r["CER"])
        agg[r["config"]]["WER"].append(r["WER"])
        agg[r["config"]]["FA"].append(r["field_accuracy"])

    for name, vals in agg.items():
        avg_cer = sum(vals["CER"]) / len(vals["CER"])
        avg_wer = sum(vals["WER"]) / len(vals["WER"])
        avg_fa  = sum(vals["FA"])  / len(vals["FA"])
        print(f"  {name:<33} {avg_cer:>7.1%}  {avg_wer:>7.1%}  {avg_fa:>9.1%}")

    print("=" * 70)

    # Lưu kết quả chi tiết
    out_file = Path(__file__).parent / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả chi tiết lưu tại: {out_file}")


if __name__ == "__main__":
    main()

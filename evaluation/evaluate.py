"""
Evaluate OCR configurations with CER, WER, field accuracy, and per-region stats.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --limit 3 --skip-gemini
    python evaluation/evaluate.py --images sample_001.jpg sample_002.jpg

Required:
    TESSERACT_CMD in .env, or tesseract available in PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent

import cv2
from dotenv import load_dotenv

try:
    import editdistance
except ModuleNotFoundError:
    editdistance = None


sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.core.data_parser import MedicalRecordParser
from src.core.detector import MedicalDetector
from src.core.reader import MedicalReader
from src.core.reader_hybrid import HybridReader
from src.utils.image_processing import deskew_image


DEFAULT_YOLO_MODEL = ROOT / "models" / "weights" / "best.pt"
GT_FILE = Path(__file__).parent / "ground_truth.json"
IMG_DIR = Path(__file__).parent / "images"
REGIONS = [
    "hospital_header",
    "patient_info",
    "diagnosis_block",
    "test_table",
    "footer_signature",
]


def compute_cer(pred: str, ref: str) -> float:
    pred = pred.strip()
    ref = ref.strip()
    if not ref:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, ref) / len(ref)


def compute_wer(pred: str, ref: str) -> float:
    pred_words = pred.strip().split()
    ref_words = ref.strip().split()
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


def normalize_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(value.strip().lower().split())


def field_match(pred_val: Optional[str], gt_val: Optional[str]) -> bool:
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    return normalize_field(pred_val) == normalize_field(gt_val)


def crop_regions(detector: MedicalDetector, img_path: Path) -> tuple[Dict[str, object], dict]:
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    image = deskew_image(image)
    detections = detector.detect(image)
    crops = {item.label: item.cropped_image for item in detections}
    detection_details = [
        {
            "label": item.label,
            "confidence": round(item.confidence, 6),
            "bbox_xyxy": list(item.bbox_xyxy),
            "crop_size": {
                "width": int(item.cropped_image.shape[1]),
                "height": int(item.cropped_image.shape[0]),
            },
        }
        for item in detections
    ]
    missing = [label for label in REGIONS if label not in crops]
    return crops, {
        "detected_count": len(detections),
        "detected_labels": sorted(crops),
        "missing_regions": missing,
        "detections": detection_details,
    }


def run_config(
    config_name: str,
    crops: Dict[str, object],
    reader_fn: Callable[[object, str], str],
    parser: MedicalRecordParser,
    gt_item: dict,
) -> dict:
    gt_regions = gt_item.get("regions", {})
    gt_fields = gt_item.get("fields", {})

    region_cer: list[float] = []
    region_wer: list[float] = []
    region_details = {}
    ocr_texts: Dict[str, str] = {}

    for label in REGIONS:
        crop = crops.get(label)
        gt_text = gt_regions.get(label, "")
        elapsed = 0.0

        if crop is None:
            pred_text = ""
        else:
            start = time.perf_counter()
            pred_text = reader_fn(crop, label)
            elapsed = time.perf_counter() - start

        ocr_texts[label] = pred_text

        cer = compute_cer(pred_text, gt_text) if gt_text else 0.0
        wer = compute_wer(pred_text, gt_text) if gt_text else 0.0
        if gt_text:
            region_cer.append(cer)
            region_wer.append(wer)

        region_details[label] = {
            "CER": cer,
            "WER": wer,
            "missing_detection": crop is None,
            "ref_length": len(gt_text),
            "pred_length": len(pred_text),
            "elapsed_sec": round(elapsed, 3),
        }

    parsed = parser.parse(ocr_texts)
    extracted = parsed.get("extractedData", {})

    correct = 0
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

    total_fields = len(gt_fields)
    return {
        "config": config_name,
        "CER": sum(region_cer) / len(region_cer) if region_cer else 0.0,
        "WER": sum(region_wer) / len(region_wer) if region_wer else 0.0,
        "field_accuracy": correct / total_fields if total_fields else 0.0,
        "region_details": region_details,
        "field_details": field_details,
    }


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def build_summary(results: list[dict], detections: dict[str, dict]) -> dict:
    by_config: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"CER": [], "WER": [], "field_accuracy": []}
    )
    by_region: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"CER": [], "WER": [], "missing": []})
    )
    field_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        config = result["config"]
        by_config[config]["CER"].append(result["CER"])
        by_config[config]["WER"].append(result["WER"])
        by_config[config]["field_accuracy"].append(result["field_accuracy"])

        for region, detail in result["region_details"].items():
            by_region[config][region]["CER"].append(detail["CER"])
            by_region[config][region]["WER"].append(detail["WER"])
            by_region[config][region]["missing"].append(int(detail["missing_detection"]))

        for field, detail in result["field_details"].items():
            field_stats[field]["correct"] += int(detail["match"])
            field_stats[field]["total"] += 1

    detection_missing: dict[str, int] = defaultdict(int)
    for detail in detections.values():
        for region in detail["missing_regions"]:
            detection_missing[region] += 1

    return {
        "images_evaluated": len(detections),
        "configs": {
            config: {
                "CER": average(vals["CER"]),
                "WER": average(vals["WER"]),
                "field_accuracy": average(vals["field_accuracy"]),
            }
            for config, vals in by_config.items()
        },
        "regions": {
            config: {
                region: {
                    "CER": average(vals["CER"]),
                    "WER": average(vals["WER"]),
                    "missing_rate": average(vals["missing"]),
                }
                for region, vals in regions.items()
            }
            for config, regions in by_region.items()
        },
        "fields": {
            field: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for field, stats in field_stats.items()
        },
        "detection_missing_counts": dict(sorted(detection_missing.items())),
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'CONFIG':<35} {'CER':>8} {'WER':>8} {'FIELD ACC':>10}")
    print("-" * 78)
    for name, vals in summary["configs"].items():
        print(
            f"{name:<35} "
            f"{vals['CER']:>7.1%}  {vals['WER']:>7.1%}  {vals['field_accuracy']:>9.1%}"
        )

    print("\nMissing detections:")
    if summary["detection_missing_counts"]:
        for region, count in summary["detection_missing_counts"].items():
            print(f"  {region:<20} {count}")
    else:
        print("  none")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR pipeline.")
    parser.add_argument("--ground-truth", type=Path, default=GT_FILE)
    parser.add_argument("--image-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(__file__).parent / "evaluation_summary.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--skip-paddle", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_model = Path(os.getenv("YOLO_MODEL_PATH") or DEFAULT_YOLO_MODEL)
    tesseract_cmd = os.getenv("TESSERACT_CMD") or "tesseract"

    print("Loading YOLO detector...")
    detector = MedicalDetector(model_path=str(yolo_model), conf_threshold=0.25)
    parser = MedicalRecordParser()

    print("Loading OCR readers...")
    tess_reader = MedicalReader(tesseract_cmd=tesseract_cmd)
    hybrid_no_gemini = None
    hybrid_gemini = None
    if not args.skip_paddle:
        hybrid_no_gemini = HybridReader(
            tesseract_cmd=tesseract_cmd,
            enable_gemini_validation=False,
        )
        hybrid_gemini = HybridReader(
            tesseract_cmd=tesseract_cmd,
            enable_gemini_validation=not args.skip_gemini,
        )

    configs = [
        ("Tesseract only", lambda img, lbl: tess_reader.read_text(img, lbl)),
    ]
    if hybrid_no_gemini is not None:
        configs.extend(
            [
                ("PaddleOCR only", lambda img, lbl: hybrid_no_gemini.read_paddle_only(img, lbl)),
                ("PaddleOCR + Tesseract", lambda img, lbl: hybrid_no_gemini.read_ocr_only(img, lbl)),
            ]
        )
    if hybrid_gemini is not None and not args.skip_gemini:
        configs.append(
            (
                "PaddleOCR + Tesseract + Gemini",
                lambda img, lbl: hybrid_gemini.read_text(img, lbl),
            )
        )

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    if args.images:
        selected = set(args.images)
        ground_truth = [item for item in ground_truth if item["image"] in selected]
    if args.limit is not None:
        ground_truth = ground_truth[: args.limit]

    all_results: list[dict] = []
    detection_results: dict[str, dict] = {}

    for gt_item in ground_truth:
        image_name = gt_item["image"]
        img_path = args.image_dir / image_name
        print(f"\n=== Image: {image_name} ===")

        try:
            crops, detection_info = crop_regions(detector, img_path)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        detection_results[image_name] = detection_info
        print(
            f"  Detected {detection_info['detected_count']} regions: "
            f"{detection_info['detected_labels']}"
        )
        if detection_info["missing_regions"]:
            print(f"  Missing regions: {detection_info['missing_regions']}")

        for config_name, reader_fn in configs:
            result = run_config(config_name, crops, reader_fn, parser, gt_item)
            result["image"] = image_name
            all_results.append(result)
            print(
                f"  [{config_name}] "
                f"CER={result['CER']:.1%}  WER={result['WER']:.1%}  "
                f"FieldAcc={result['field_accuracy']:.1%}"
            )

    warnings = []
    for reader_name, reader in (
        ("PaddleOCR no Gemini", hybrid_no_gemini),
        ("PaddleOCR with Gemini", hybrid_gemini),
    ):
        if reader is not None and reader.last_paddle_error:
            warnings.append(f"{reader_name} failed: {reader.last_paddle_error}")

    summary = build_summary(all_results, detection_results)
    summary["warnings"] = warnings
    print_summary(summary)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    payload = {
        "metadata": {
            "ground_truth": str(args.ground_truth),
            "image_dir": str(args.image_dir),
            "yolo_model": str(yolo_model),
            "tesseract_cmd": tesseract_cmd,
            "paddle_enabled": not args.skip_paddle,
            "gemini_enabled": not args.skip_gemini,
            "warnings": warnings,
        },
        "detections": detection_results,
        "results": all_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results: {args.output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()

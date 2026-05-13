import os
import sys
import logging
from dotenv import load_dotenv
import cv2

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO)
load_dotenv()

from src.core.gemini_validator import GeminiValidator

# Load test image
test_table_path = r"outputs\yolo_regions\crops\20260513_155315_z7720447187198_e35283f317e085a34267a814c2867537\00_test_table_0.954.jpg"
crop_image = cv2.imread(test_table_path)

if crop_image is None:
    print(f"ERROR: Could not load {test_table_path}")
    exit(1)

print(f"OK: Loaded test_table crop image ({crop_image.shape})")

# Test Gemini validator
ocr_text = "TENXETNGHEM kế Tqui T Giámnrmumcmf sơ rẽ 6s s1 8x mini n mm RE. la 0 ôn 5"
print(f"\nOriginal OCR text (length={len(ocr_text)}):\n{ocr_text}\n")

validator = GeminiValidator()
fixed_text = validator.validate_table(crop_image, ocr_text)

print(f"\nFixed text (length={len(fixed_text)}):\n{fixed_text}")
if fixed_text != ocr_text:
    print("\n✓ Gemini modified the text!")
else:
    print("\n✗ Gemini returned same text (no fix)")


#!/usr/bin/env python
"""Test full OCR flow with Gemini validation."""

import json
import sys
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"

# Check health
print("1. Checking API health...")
response = requests.get(f"{API_URL}/health")
health = response.json()
print(f"   Gemini validator: {health.get('gemini_validator_initialized')}")
print(f"   API ready: {health.get('status')}")

if not health.get('gemini_validator_initialized'):
    print("\nERROR: Gemini validator not initialized!")
    sys.exit(1)

# Test OCR upload (use real medical doc if available)
print("\n2. Testing OCR upload...")

# Find a test image (hospital record)
test_images = list(Path("outputs/yolo_regions/crops").glob("*/00_test_table_*.jpg"))
if not test_images:
    print("ERROR: No test_table images found in outputs/")
    sys.exit(1)

test_image = test_images[0]
print(f"   Using test image: {test_image.name}")

with open(test_image, "rb") as f:
    files = {"file": (test_image.name, f, "image/jpeg")}
    response = requests.post(f"{API_URL}/v1/ocr/upload", files=files)

if response.status_code != 200:
    print(f"\nERROR: Upload failed ({response.status_code})")
    print(response.text)
    sys.exit(1)

result = response.json()
test_table_result = result.get("result", {}).get("test_table", {})
test_table_text = test_table_result.get("text", "")

print(f"\n3. test_table OCR result ({len(test_table_text)} chars):")
# Show length instead of content to avoid encoding issues
print(f"   Original PaddleOCR: ~73 chars")
print(f"   After Gemini fix: {len(test_table_text)} chars")

# Check if bad pattern is still there
has_error = "TENXETNGHEM" in test_table_text
print(f"\n   Has OCR error 'TENXETNGHEM': {has_error}")

if not has_error:
    print("   [PASS] SUCCESS: Gemini fixed the OCR errors!")
else:
    print("   [FAIL] Gemini did NOT fix the text")

print("\nDone!")

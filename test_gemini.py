#!/usr/bin/env python
"""Test Gemini Vision API validation for OCR results."""

import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.gemini_validator import GeminiValidator


def test_gemini_api():
    """Test Gemini API connection & basic validation."""
    print("=" * 60)
    print("Testing Gemini Vision API Setup")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        print("Please run: python test_gemini.py after setting up SETUP_GEMINI.md")
        return False

    print(f"OK: API key found ({len(api_key)} chars)")

    # Try to init validator
    try:
        validator = GeminiValidator(api_key=api_key)
        print("OK: GeminiValidator initialized")
    except Exception as e:
        print(f"ERROR: Failed to init GeminiValidator: {e}")
        return False

    # Create test image (simple table-like structure)
    print("\nCreating test image...")
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add some text-like patterns
    cv2.rectangle(test_image, (10, 10), (390, 190), (0, 0, 0), 2)
    cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Test validation with better error handling
    print("\nTesting Gemini Vision API...")
    test_ocr_text = "TENXETNGHEM keTqui Test"
    try:
        # Call Gemini directly to see response
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            [
                {"mime_type": "image/png", "data": GeminiValidator._encode_image(test_image)},
                "Describe what you see in this image.",
            ]
        )
        print(f"Gemini Response: {response.text[:100] if response.text else 'NO TEXT'}")

        fixed_text = validator.validate_text(
            test_image,
            test_ocr_text,
            label="test_table",
            fallback_text="FALLBACK TEXT",
        )
        print(f"Input:  {test_ocr_text}")
        print(f"Output: {fixed_text}")
        print("\nOK: Gemini validation working!")
        return True
    except Exception as e:
        print(f"ERROR: Gemini validation failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gemini_api()
    sys.exit(0 if success else 1)

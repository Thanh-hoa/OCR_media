from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Optional
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GeminiValidator:
    """Dùng Gemini Vision API để fix OCR errors & validate text từ crop image."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY không được set. "
                "Hãy set environment variable hoặc pass api_key directly."
            )
        self.timeout = timeout
        self._client = None

    def _init_client(self):
        """Lazy init Gemini client."""
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai

    def validate_text(
        self,
        crop_image: np.ndarray,
        ocr_text: str,
        label: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        """
        Dùng Gemini Vision API để validate & fix OCR text.

        Args:
            crop_image: numpy array (BGR)
            ocr_text: text từ PaddleOCR/Tesseract (có thể sai)
            label: class name (test_table, hospital_header, etc.)
            fallback_text: text để fallback nếu API fail (Tesseract result)

        Returns:
            Fixed text từ Gemini hoặc fallback_text nếu API fail
        """
        try:
            self._init_client()
            image_b64 = self._encode_image(crop_image)
            prompt = self._build_prompt(ocr_text, label)
            response = self._client.GenerativeModel("gemini-2.5-flash").generate_content(
                [
                    {"mime_type": "image/png", "data": image_b64},
                    prompt,
                ]
            )
            fixed_text = response.text.strip() if response.text else ocr_text
            logger.info(f"Gemini validate_text ({label}): {len(ocr_text)} → {len(fixed_text)} chars")
            return fixed_text if fixed_text else fallback_text
        except Exception as e:
            logger.error(f"Gemini validate_text ({label}) failed: {e}")
            return fallback_text if fallback_text else ocr_text

    def validate_table(self, crop_image: np.ndarray, ocr_text: str) -> str:
        """Specialized validation cho test_table."""
        try:
            self._init_client()
            image_b64 = self._encode_image(crop_image)
            prompt = (
                "This is a medical test table. OCR output:\n"
                f"{ocr_text}\n\n"
                "Fix any OCR errors (e.g., 'TENXETNGHEM' → 'TEN XET NGHEM'). "
                "Preserve table structure with | for column separators and \\n for rows. "
                "Output ONLY the corrected table text, no explanation."
            )
            response = self._client.GenerativeModel("gemini-2.5-flash").generate_content(
                [
                    {"mime_type": "image/png", "data": image_b64},
                    prompt,
                ]
            )
            fixed_text = response.text.strip() if response.text else ocr_text
            logger.info(f"Gemini validate_table: {len(ocr_text)} → {len(fixed_text)} chars")
            return fixed_text
        except Exception as e:
            logger.error(f"Gemini validate_table failed: {e}")
            return ocr_text

    @staticmethod
    def _encode_image(image_bgr: np.ndarray) -> str:
        """Convert BGR numpy array → base64 PNG string."""
        _, buffer = cv2.imencode(".png", image_bgr)
        return base64.standard_b64encode(buffer).decode("utf-8")

    @staticmethod
    def _build_prompt(ocr_text: str, label: Optional[str]) -> str:
        """Build appropriate prompt based on label."""
        if label == "test_table":
            return (
                "This is a medical test table. Fix OCR errors in this text, "
                "preserving structure with | for columns and \\n for rows. "
                "Output ONLY the corrected text:\n" + ocr_text
            )

        if label in {"hospital_header", "patient_info"}:
            return (
                "This is medical document text. Fix OCR errors while preserving meaning. "
                "Output ONLY the corrected text:\n" + ocr_text
            )

        return (
            "Fix OCR errors in this text. Output ONLY the corrected text:\n" + ocr_text
        )

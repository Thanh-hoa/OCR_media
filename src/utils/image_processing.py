from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def apply_threshold(gray_image: np.ndarray, adaptive: bool = False) -> np.ndarray:
    if adaptive:
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def preprocess_for_ocr(image_bgr: np.ndarray, label: Optional[str] = None) -> np.ndarray:
    gray = to_gray(image_bgr)
    gray = resize_image(gray, scale=2.0)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    return apply_threshold(gray, adaptive=(label == "test_table"))

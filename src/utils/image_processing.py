from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def deskew_image(image_bgr: np.ndarray, max_skew_deg: float = 10.0) -> np.ndarray:
    """Tự động xoay thẳng ảnh bị nghiêng nhẹ (< max_skew_deg độ).

    Thuật toán:
      1. Invert + Otsu threshold → text thành pixel trắng trên nền đen
      2. Dilation ngang → nối ký tự thành các dòng chữ dài
      3. findContours → minAreaRect trên từng dòng → lấy góc nghiêng
      4. Median của tất cả góc → loại bỏ nhiễu từ đường kẻ bảng, chữ ký...
      5. Chỉ xoay nếu 0.5° ≤ |skew| ≤ max_skew_deg (tránh over-correct)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = image_bgr.shape[:2]
    # Kernel ngang dài để nối các chữ thành dòng, nhưng không nối các hàng với nhau
    kw = max(20, w // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    min_area = w * h * 0.0005  # bỏ qua contour quá nhỏ (nhiễu, dấu chấm...)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        angle = cv2.minAreaRect(cnt)[-1]
        # OpenCV trả angle trong (-90, 0]. Chuẩn hoá về (-45, 45]:
        if angle < -45:
            angle += 90
        if abs(angle) <= max_skew_deg:
            angles.append(angle)

    if not angles:
        return image_bgr

    skew = float(np.median(angles))
    if abs(skew) < 0.5:
        return image_bgr

    center = (w // 2, h // 2)
    # Xoay ngược chiều nghiêng để thẳng lại
    M = cv2.getRotationMatrix2D(center, -skew, 1.0)
    return cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


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

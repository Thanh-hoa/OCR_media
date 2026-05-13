from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract

from src.core.gemini_validator import GeminiValidator
from src.utils.image_processing import preprocess_for_ocr


class HybridReader:
    def __init__(
        self,
        tesseract_cmd: str,
        lang_paddle: str = "vi",
        lang_tesseract: str = "vie",
        confidence_threshold: float = 0.5,
        enable_gemini_validation: bool = True,
    ) -> None:
        self.ocr_paddle = None
        self.lang_paddle = lang_paddle

        exe_path = Path(tesseract_cmd)
        if not exe_path.exists():
            raise FileNotFoundError(f"Tesseract not found: {exe_path}")
        pytesseract.pytesseract.tesseract_cmd = str(exe_path)

        self.lang_tesseract = lang_tesseract
        self.confidence_threshold = confidence_threshold
        self.oem = 3

        self.label_conf_threshold = {
            "hospital_header": 0.55,
            "patient_info": 0.5,
            "diagnosis_block": 0.5,
            "test_table": 0.45,
            "footer_signature": 0.35,
        }

        # Init Gemini validator (optional, nếu có API key)
        self.gemini_validator = None
        if enable_gemini_validation and os.getenv("GEMINI_API_KEY"):
            try:
                self.gemini_validator = GeminiValidator()
            except ValueError:
                pass  # API key not set, disable validation

    def _init_paddle(self) -> None:
        if self.ocr_paddle is None:
            self.ocr_paddle = PaddleOCR(use_angle_cls=True, lang=self.lang_paddle)

    def read_text(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        if image_bgr is None or image_bgr.size == 0:
            return ""

        candidates = self._generate_candidates(image_bgr, label)

        paddle_best_text = ""
        paddle_best_conf = 0.0
        for candidate in candidates:
            text_paddle, conf_paddle = self._read_paddle(candidate, label=label)
            if self._is_better_candidate(
                text_paddle,
                conf_paddle,
                paddle_best_text,
                paddle_best_conf,
            ):
                paddle_best_text = text_paddle
                paddle_best_conf = conf_paddle

        tesseract_best_text = ""
        tesseract_best_score = -1.0
        for candidate in candidates:
            text_tesseract = self._read_tesseract(candidate, label)
            score = self._text_quality_score(text_tesseract)
            if score > tesseract_best_score:
                tesseract_best_text = text_tesseract
                tesseract_best_score = score

        required_conf = self.label_conf_threshold.get(label or "", self.confidence_threshold)
        if label == "test_table" and paddle_best_text:
            # For table regions, Paddle usually keeps structure better than Tesseract.
            if self._text_quality_score(paddle_best_text) + 15.0 >= self._text_quality_score(tesseract_best_text):
                final_text = paddle_best_text
            else:
                final_text = tesseract_best_text if tesseract_best_text else paddle_best_text
        elif paddle_best_text and paddle_best_conf >= required_conf:
            final_text = paddle_best_text
        elif self._text_quality_score(tesseract_best_text) > self._text_quality_score(paddle_best_text):
            final_text = tesseract_best_text
        else:
            final_text = paddle_best_text

        # Validate & fix with Gemini (optional, nếu có API key)
        if self.gemini_validator and final_text:
            if label == "test_table":
                final_text = self.gemini_validator.validate_table(image_bgr, final_text)
            else:
                final_text = self.gemini_validator.validate_text(
                    image_bgr, final_text, label=label, fallback_text=final_text
                )

        return final_text

    def _generate_candidates(self, image_bgr: np.ndarray, label: Optional[str]) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []

        base = preprocess_for_ocr(image_bgr, label=label)
        candidates.append(base)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(otsu)

        if label in {"hospital_header", "patient_info", "diagnosis_block", "footer_signature"}:
            adaptive = cv2.adaptiveThreshold(
                clahe,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )
            candidates.append(adaptive)

        if label == "footer_signature":
            inverted = cv2.bitwise_not(otsu)
            candidates.append(inverted)

        return candidates

    def _read_paddle(self, image_bgr: np.ndarray, label: Optional[str] = None) -> tuple[str, float]:
        self._init_paddle()
        try:
            result = self.ocr_paddle.ocr(image_bgr, cls=True)
            
            if not result or not result[0]:
                return "", 0.0
            
            texts = []
            confidences = []
            rows = []
            
            for line in result[0]:
                if not line or len(line) < 2 or not line[1]:
                    continue

                box = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                if confidence <= 0.3 or not text:
                    continue

                avg_y = sum(point[1] for point in box) / 4.0
                avg_x = sum(point[0] for point in box) / 4.0
                rows.append((avg_y, avg_x, text, confidence))

            rows.sort(key=lambda r: (r[0], r[1]))

            for _, _, text, confidence in rows:
                texts.append(text)
                confidences.append(confidence)
            
            if not texts:
                return "", 0.0
            
            if label == "test_table":
                full_text = self._format_table_rows(rows, image_bgr.shape[1])
            elif label in {"hospital_header", "diagnosis_block"}:
                full_text = self._format_two_column(rows, image_bgr.shape[1])
            else:
                full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return self._normalize_text(full_text, label=label), avg_confidence
            
        except Exception as e:
            return "", 0.0

    def _read_tesseract(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        try:
            config = self._build_config(label)
            text = pytesseract.image_to_string(image_bgr, lang=self.lang_tesseract, config=config)
            return self._normalize_text(text)
        except Exception as e:
            return ""

    def _build_config(self, label: Optional[str]) -> str:
        psm_map = {
            "hospital_header": 11,  # ← Changed from 5 to 11 (sparse text with layout)
            "patient_info": 6,
            "diagnosis_block": 6,
            "test_table": 6,
            "footer_signature": 7,
        }
        psm = psm_map.get(label, 6)
        return f"--oem {self.oem} --psm {psm} -c preserve_interword_spaces=1"

    def _is_better_candidate(
        self,
        new_text: str,
        new_conf: float,
        old_text: str,
        old_conf: float,
    ) -> bool:
        if not new_text:
            return False
        if not old_text:
            return True
        if new_conf > old_conf + 1e-9:
            return True
        if abs(new_conf - old_conf) <= 1e-9:
            return self._text_quality_score(new_text) > self._text_quality_score(old_text)
        return False

    @staticmethod
    def _text_quality_score(text: str) -> float:
        if not text:
            return 0.0

        valid_chars = sum(ch.isalnum() or ch.isspace() or ch in "-/:.,()%" for ch in text)
        ratio = valid_chars / max(1, len(text))
        long_tokens = sum(1 for token in text.split() if len(token) >= 2)
        return ratio * 100.0 + min(long_tokens, 50)

    @staticmethod
    def _format_two_column(rows: list[tuple[float, float, str, float]], image_width: int) -> str:
        """Tách text của vùng có 2 cột (trái / phải) thay vì trộn lẫn theo Y.

        Ví dụ hospital_header:
          Cột trái  — tên bệnh viện, địa chỉ, SĐT
          Cột phải  — PID, số bệnh phẩm, mã bệnh án

        Ngưỡng phân cột: 55% chiều rộng ảnh crop để ưu tiên cột trái rộng hơn.
        Output: "<left_text> | <right_text>" hoặc chỉ một cột nếu cột kia trống.
        """
        mid_x = image_width * 0.55
        left_rows  = [(y, x, t) for y, x, t, _ in rows if x <= mid_x]
        right_rows = [(y, x, t) for y, x, t, _ in rows if x > mid_x]

        left_text  = " ".join(t for _, _, t in sorted(left_rows,  key=lambda r: r[0]))
        right_text = " ".join(t for _, _, t in sorted(right_rows, key=lambda r: r[0]))

        parts = [p.strip() for p in (left_text, right_text) if p.strip()]
        return " | ".join(parts)

    @staticmethod
    def _format_table_rows(rows: list[tuple[float, float, str, float]], image_width: int) -> str:
        if not rows:
            return ""

        row_threshold = 24.0
        grouped: list[list[tuple[float, float, str, float]]] = []

        for item in rows:
            if not grouped:
                grouped.append([item])
                continue

            prev_y = sum(r[0] for r in grouped[-1]) / len(grouped[-1])
            if abs(item[0] - prev_y) <= row_threshold:
                grouped[-1].append(item)
            else:
                grouped.append([item])

        lines: list[str] = []
        gap_threshold = max(80.0, image_width * 0.08)

        for row_items in grouped:
            row_sorted = sorted(row_items, key=lambda r: r[1])
            row_parts: list[str] = []
            prev_x: Optional[float] = None
            for _, x, text, _ in row_sorted:
                token = text.strip()
                if not token:
                    continue
                if prev_x is not None and (x - prev_x) > gap_threshold and row_parts:
                    row_parts.append("|")
                row_parts.append(token)
                prev_x = x

            row_text = " ".join(row_parts).strip()
            if row_text:
                lines.append(row_text)

        return "\n".join(lines)

    @staticmethod
    def _normalize_text(text: str, label: Optional[str] = None) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        if label == "test_table":
            text = text.replace("\r", "\n")
            text = re.sub(r"\n+", "\n", text)
            lines = []
            for line in text.split("\n"):
                line = re.sub(r"\s+", " ", line).strip()
                line = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\-/:.,()%| ]", "", line)
                line = re.sub(r"([:.,%/\-|])\1+", r"\1", line)
                if line:
                    lines.append(line)
            return "\n".join(lines)

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        cleaned_tokens = []
        for token in text.split(" "):
            token = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\-/:.,()%]", "", token)
            if not token:
                continue

            letter_count = sum(ch.isalpha() for ch in token)
            digit_count = sum(ch.isdigit() for ch in token)
            punct_count = sum(ch in "-/:.,()%" for ch in token)
            if letter_count + digit_count == 0:
                continue
            if punct_count > max(letter_count + digit_count, 1):
                continue

            cleaned_tokens.append(token)

        text = " ".join(cleaned_tokens).strip()
        text = re.sub(r"([:.,%/\-])\1+", r"\1", text)
        return text


MedicalReaderHybrid = HybridReader

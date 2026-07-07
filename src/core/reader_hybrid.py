from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import unicodedata
import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_TMP = PROJECT_ROOT / ".tmp"
WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(WORKSPACE_TMP / "paddlex_cache"))
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract

from src.core.gemini_validator import GeminiValidator
from src.i18n.message_translator import message_translator
from src.utils.image_processing import preprocess_for_ocr

logger = logging.getLogger(__name__)


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
        self._paddle_unavailable = False
        self.last_paddle_error: Optional[str] = None
        self._paddle_lock = threading.Lock()
        self.lang_paddle = lang_paddle

        resolved_tesseract_cmd = self._resolve_tesseract_cmd(tesseract_cmd)
        pytesseract.pytesseract.tesseract_cmd = resolved_tesseract_cmd

        self.lang_tesseract = lang_tesseract
        self.confidence_threshold = confidence_threshold
        self.oem = 3

        # Init Gemini validator (optional, nếu có API key)
        self.gemini_validator = None
        if enable_gemini_validation and os.getenv("GEMINI_API_KEY"):
            try:
                self.gemini_validator = GeminiValidator()
                logger.info("GeminiValidator initialized successfully")
            except ValueError as e:
                logger.warning(f"GeminiValidator init failed: {e}")
        else:
            if not enable_gemini_validation:
                logger.debug("Gemini validation disabled by parameter")
            else:
                logger.debug("GEMINI_API_KEY not set, skipping GeminiValidator")

    def _init_paddle(self) -> None:
        if self.ocr_paddle is None:
            with self._paddle_lock:
                if self.ocr_paddle is None:
                    old_tempdir = tempfile.tempdir
                    old_env = {
                        name: os.environ.get(name)
                        for name in ("TMP", "TEMP", "TMPDIR")
                    }
                    try:
                        os.environ["TMP"] = str(WORKSPACE_TMP)
                        os.environ["TEMP"] = str(WORKSPACE_TMP)
                        os.environ["TMPDIR"] = str(WORKSPACE_TMP)
                        tempfile.tempdir = str(WORKSPACE_TMP)
                        self.ocr_paddle = PaddleOCR(
                            lang=self.lang_paddle,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                        )
                    finally:
                        tempfile.tempdir = old_tempdir
                        for name, value in old_env.items():
                            if value is None:
                                os.environ.pop(name, None)
                            else:
                                os.environ[name] = value

    @staticmethod
    def _resolve_tesseract_cmd(tesseract_cmd: str) -> str:
        cmd = str(tesseract_cmd).strip()
        if not cmd:
            raise FileNotFoundError(message_translator.get_message("tesseract.empty_cmd"))

        cmd_path = Path(cmd)
        has_path_separator = any(separator in cmd for separator in ("\\", "/"))
        if has_path_separator or cmd_path.is_absolute():
            if cmd_path.exists():
                return str(cmd_path)
            raise FileNotFoundError(
                message_translator.get_message("tesseract.not_found", path=str(cmd_path))
            )

        resolved = shutil.which(cmd)
        if resolved:
            return resolved
        raise FileNotFoundError(
            message_translator.get_message("tesseract.not_found_in_path", cmd=cmd)
        )

    def read_paddle_only(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        """Cấu hình 2: PaddleOCR only, không fallback, không Gemini."""
        if image_bgr is None or image_bgr.size == 0:
            return ""
        prep = preprocess_for_ocr(image_bgr, label=label)
        text, _ = self._read_paddle(prep, label=label)
        return text

    def read_ocr_only(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        """Cấu hình 3: PaddleOCR + Tesseract fallback, không Gemini."""
        if image_bgr is None or image_bgr.size == 0:
            return ""
        if label == "test_table":
            return self._read_table_tesseract(image_bgr)
        prep = preprocess_for_ocr(image_bgr, label=label)
        text, conf = self._read_paddle(prep, label=label)
        if not text or conf < 0.3:
            fallback = self._read_tesseract(prep, label)
            return fallback if fallback else text
        return text

    def read_text(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        if image_bgr is None or image_bgr.size == 0:
            return ""

        if label == "test_table":
            final_text = self._read_table_tesseract(image_bgr)
            if self.gemini_validator and final_text:
                return self.gemini_validator.validate_table(image_bgr, final_text)
            return final_text

        # Optimize: Use ONLY the best preprocessing (preprocess_for_ocr)
        prep_image = preprocess_for_ocr(image_bgr, label=label)

        # Step 1: Try PaddleOCR (primary, faster & more accurate)
        text_paddle, conf_paddle = self._read_paddle(prep_image, label=label)
        logger.debug(f"PaddleOCR ({label}): conf={conf_paddle:.3f}, len={len(text_paddle)}")

        # Step 2: Fallback to Tesseract ONLY if PaddleOCR fails
        if not text_paddle or conf_paddle < 0.3:
            logger.debug(f"PaddleOCR confidence too low ({conf_paddle}), trying Tesseract...")
            text_tesseract = self._read_tesseract(prep_image, label)
            final_text = text_tesseract if text_tesseract else text_paddle
        else:
            final_text = text_paddle

        # Step 3: Validate & fix with Gemini for ALL regions (especially for noise cleanup)
        if self.gemini_validator and final_text:
            logger.debug(f"Calling Gemini validator for {label}...")
            if label == "test_table":
                final_text = self.gemini_validator.validate_table(image_bgr, final_text)
            else:
                # Use validate_text for other regions (with custom prompts)
                final_text = self.gemini_validator.validate_text(
                    image_bgr, final_text, label=label, fallback_text=final_text
                )
            logger.info(f"Gemini fixed {label}: {len(text_paddle)} → {len(final_text)} chars")
        else:
            if not self.gemini_validator:
                logger.warning("Gemini validator not available — trả về OCR thô (kiểm tra GEMINI_API_KEY)")

        return final_text


    def _read_paddle(self, image_bgr: np.ndarray, label: Optional[str] = None) -> tuple[str, float]:
        if self._paddle_unavailable:
            return "", 0.0
        try:
            self._init_paddle()
            result = self.ocr_paddle.predict(image_bgr)
            
            if not result:
                return "", 0.0
            
            texts = []
            confidences = []
            rows = self._extract_paddle_rows(result)

            rows.sort(key=lambda r: (r[0], r[1]))

            for _, _, text, confidence in rows:
                texts.append(text)
                confidences.append(confidence)
            
            if not texts:
                return "", 0.0
            
            if label == "test_table":
                full_text = self._format_table_rows(rows, image_bgr.shape[1])
            elif label == "hospital_header":
                # Chỉ lấy cột trái (tên BV, địa chỉ, ĐT) — bỏ cột phải (PID, KHOA, Mã bệnh án)
                mid_x = image_bgr.shape[1] * 0.55
                left_rows = [(y, x, t, c) for y, x, t, c in rows if x <= mid_x]
                full_text = "\n".join(t for _, _, t, _ in sorted(left_rows, key=lambda r: r[0]))
            elif label == "diagnosis_block":
                full_text = self._format_two_column(rows, image_bgr.shape[1])
            elif label in {"patient_info", "footer_signature"}:
                full_text = self._format_table_rows(rows, image_bgr.shape[1])
            else:
                full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences)
            
            return self._normalize_text(full_text, label=label), avg_confidence
            
        except Exception as e:
            self._paddle_unavailable = True
            self.last_paddle_error = str(e)
            logger.warning("PaddleOCR failed for %s: %s", label, e)
            return "", 0.0

    @staticmethod
    def _extract_paddle_rows(result) -> list[tuple[float, float, str, float]]:
        rows: list[tuple[float, float, str, float]] = []

        # PaddleOCR 3.x returns a list of dict-like OCRResult objects.
        for page in result:
            if hasattr(page, "get") and page.get("rec_texts") is not None:
                texts = page.get("rec_texts")
                scores = page.get("rec_scores")
                boxes = page.get("rec_polys")
                if boxes is None:
                    boxes = page.get("dt_polys")
                texts = [] if texts is None else texts
                scores = [] if scores is None else scores
                boxes = [] if boxes is None else boxes
                for idx, text in enumerate(texts):
                    confidence = float(scores[idx]) if idx < len(scores) else 0.0
                    if confidence <= 0.3 or not text:
                        continue
                    box = boxes[idx] if idx < len(boxes) else []
                    avg_y, avg_x = HybridReader._box_center(box)
                    rows.append((avg_y, avg_x, str(text), confidence))
                continue

            # PaddleOCR 2.x returned [[box, [text, confidence]], ...].
            lines = page if isinstance(page, list) else []
            for line in lines:
                if not line or len(line) < 2 or not line[1]:
                    continue
                box = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                if confidence <= 0.3 or not text:
                    continue
                avg_y, avg_x = HybridReader._box_center(box)
                rows.append((avg_y, avg_x, text, confidence))

        return rows

    @staticmethod
    def _box_center(box) -> tuple[float, float]:
        if box is None or len(box) == 0:
            return 0.0, 0.0

        points = np.asarray(box, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return 0.0, 0.0
        avg_x = float(points[:, 0].mean())
        avg_y = float(points[:, 1].mean())
        return avg_y, avg_x

    def _read_tesseract(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        try:
            config = self._build_config(label)
            text = pytesseract.image_to_string(image_bgr, lang=self.lang_tesseract, config=config)
            return self._normalize_text(text)
        except Exception as e:
            return ""

    def _read_table_tesseract(self, image_bgr: np.ndarray) -> str:
        try:
            cell_text = self._read_table_by_grid(image_bgr)
            if cell_text:
                return self._normalize_text(cell_text, label="test_table")

            cleaned = self._preprocess_table_without_grid(image_bgr)
            positioned_text = self._read_table_by_positions(cleaned)
            if positioned_text:
                return self._normalize_text(positioned_text, label="test_table")

            config = (
                f"--oem {self.oem} --psm 11 "
                "-c preserve_interword_spaces=1 "
                "-c tessedit_char_blacklist=~`^_=[]{}<>"
            )
            text = pytesseract.image_to_string(
                cleaned,
                lang=self.lang_tesseract,
                config=config,
            )
            return self._normalize_text(text, label="test_table")
        except Exception as e:
            logger.warning("Table OCR failed: %s", e)
            return ""

    def _read_table_by_grid(self, image_bgr: np.ndarray) -> str:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scale = 2.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)

        binary_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
        h, w = binary_inv.shape

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(20, w // 40), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(20, h // 60))
        )
        horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel)

        xs = self._line_centers(vertical, axis="x", min_length=h * 0.08)
        ys = self._line_centers(horizontal, axis="y", min_length=w * 0.20)
        if len(xs) < 4 or len(ys) < 4:
            return ""

        xs = self._merge_positions(xs, max(8, int(w * 0.006)))
        ys = self._merge_positions(ys, max(8, int(h * 0.006)))
        if len(xs) < 4 or len(ys) < 4:
            return ""

        xs = self._select_table_columns(xs, w)
        if len(xs) < 4:
            return ""

        grid_mask = cv2.bitwise_or(horizontal, vertical)
        grid_mask = cv2.dilate(
            grid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        cleaned = cv2.inpaint(gray, grid_mask, 3, cv2.INPAINT_TELEA)
        cleaned = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        data = pytesseract.image_to_data(
            cleaned,
            lang=self.lang_tesseract,
            config=(
                f"--oem {self.oem} --psm 6 "
                "-c preserve_interword_spaces=1 "
                "-c tessedit_char_blacklist=~`^_=[]{}<>"
            ),
            output_type=pytesseract.Output.DICT,
        )

        cells: dict[tuple[int, int], list[tuple[int, str]]] = {}
        n = len(data.get("text", []))
        for i in range(n):
            token = (data["text"][i] or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 20:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            bw = int(data["width"][i])
            bh = int(data["height"][i])
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            row_idx = self._interval_index(ys, cy)
            col_idx = self._interval_index(xs, cx)
            if row_idx is None or col_idx is None:
                continue
            cells.setdefault((row_idx, col_idx), []).append((x, token))

        lines: list[str] = []
        for row_idx in range(len(ys) - 1):
            row_values: list[str] = []
            for col_idx in range(len(xs) - 1):
                tokens = cells.get((row_idx, col_idx), [])
                text = " ".join(t for _, t in sorted(tokens, key=lambda item: item[0])).strip()
                row_values.append(text)

            if len(row_values) < 4:
                continue

            name = row_values[0]
            value = row_values[1]
            reference = row_values[2]
            unit = row_values[3]

            if not name or not re.search(r"\d", value):
                continue
            if len(name) <= 1 and not re.search(r"[A-Za-zÀ-ỹà-ỹĐđ]", name):
                continue

            lines.append(f"{name} | {value} | {unit} | {reference}")

        return "\n".join(lines) if len(lines) >= 8 else ""

    def _read_table_by_positions(self, cleaned_gray: np.ndarray) -> str:
        data = pytesseract.image_to_data(
            cleaned_gray,
            lang=self.lang_tesseract,
            config=(
                f"--oem {self.oem} --psm 6 "
                "-c preserve_interword_spaces=1 "
                "-c tessedit_char_blacklist=~`^_=[]{}<>"
            ),
            output_type=pytesseract.Output.DICT,
        )

        w = cleaned_gray.shape[1]
        tokens: list[tuple[int, int, int, str]] = []
        for i, raw_token in enumerate(data.get("text", [])):
            token = (raw_token or "").strip()
            if not token:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 10:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            bw = int(data["width"][i])
            bh = int(data["height"][i])
            tokens.append((x + bw // 2, y + bh // 2, max(1, bh), token))

        if not tokens:
            return ""

        tokens.sort(key=lambda item: (item[1], item[0]))
        median_h = float(np.median([height for _, _, height, _ in tokens]))
        row_gap = max(14.0, median_h * 0.85)
        rows: list[list[tuple[int, int, str]]] = []
        row_centers: list[float] = []

        for x, y, _, token in tokens:
            if not rows or abs(y - row_centers[-1]) > row_gap:
                rows.append([(x, y, token)])
                row_centers.append(float(y))
                continue
            rows[-1].append((x, y, token))
            row_centers[-1] = sum(item[1] for item in rows[-1]) / len(rows[-1])

        lines: list[str] = []
        for row in rows:
            row = sorted(row, key=lambda item: item[0])
            cols: list[list[tuple[int, str]]] = [[], [], [], []]
            for x, _, token in row:
                if x < w * 0.40:
                    col_idx = 0
                elif x < w * 0.56:
                    col_idx = 1
                elif x < w * 0.72:
                    col_idx = 2
                else:
                    col_idx = 3
                cols[col_idx].append((x, token))

            values = [
                " ".join(token for _, token in sorted(col, key=lambda item: item[0])).strip()
                for col in cols
            ]
            name, value, unit, reference = values
            if not name or not re.search(r"\d", value):
                continue
            if re.search(r"T[EÊ]N|X[EÉ]T|NGHI|K[EÊ]T|QU[AẢ]|GI[AÁ]", name, re.I):
                continue
            lines.append(f"{name} | {value} | {unit} | {reference}")

        return "\n".join(lines) if len(lines) >= 3 else ""

    @staticmethod
    def _line_centers(mask: np.ndarray, axis: str, min_length: float) -> list[int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: list[int] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if axis == "x":
                if h < min_length:
                    continue
                centers.append(x + w // 2)
            else:
                if w < min_length:
                    continue
                centers.append(y + h // 2)
        return sorted(centers)

    @staticmethod
    def _merge_positions(values: list[int], tolerance: int) -> list[int]:
        if not values:
            return []
        values = sorted(values)
        groups: list[list[int]] = [[values[0]]]
        for value in values[1:]:
            if abs(value - groups[-1][-1]) <= tolerance:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [int(round(sum(group) / len(group))) for group in groups]

    @staticmethod
    def _select_table_columns(xs: list[int], image_width: int) -> list[int]:
        min_gap = max(20, int(image_width * 0.07))
        separated: list[int] = []
        for x in sorted(xs):
            if not separated or x - separated[-1] >= min_gap:
                separated.append(x)
            elif abs(x - separated[-1]) < min_gap:
                separated[-1] = int(round((separated[-1] + x) / 2))
        xs = separated

        if len(xs) <= 5:
            return xs

        best: list[int] = xs
        best_score = -1.0
        for start in range(0, len(xs) - 3):
            candidate = xs[start : start + 5]
            width = candidate[-1] - candidate[0]
            if width < image_width * 0.55:
                continue
            gaps = np.diff(candidate)
            if min(gaps) <= 0:
                continue
            balance = min(gaps) / max(gaps)
            score = width * (0.6 + balance)
            if score > best_score:
                best = candidate
                best_score = score
        return best

    @staticmethod
    def _interval_index(lines: list[int], value: float) -> Optional[int]:
        for idx in range(len(lines) - 1):
            if lines[idx] <= value <= lines[idx + 1]:
                return idx
        return None

    @staticmethod
    def _preprocess_table_without_grid(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)

        binary_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
        h, w = binary_inv.shape
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(40, w // 28), 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(25, h // 28))
        )
        horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel)
        grid_mask = cv2.bitwise_or(horizontal, vertical)
        grid_mask = cv2.dilate(
            grid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

        connected_grid = cv2.dilate(
            grid_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            connected_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
            pad_x = max(8, int(bw * 0.01))
            pad_y = max(4, int(bh * 0.005))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            gray = gray[y1:y2, x1:x2]
            grid_mask = grid_mask[y1:y2, x1:x2]

        cleaned = cv2.inpaint(gray, grid_mask, 3, cv2.INPAINT_TELEA)
        return cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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

        # C\u1eaft b\u1ecf k\u00fd t\u1ef1 r\u00e1c tr\u01b0\u1edbc nh\u00e3n quan tr\u1ecdng (fallback khi Gemini kh\u00f4ng ch\u1ea1y)
        if label == "patient_info":
            m = re.search(r"H\u1ecd\s*t\u00ean\s*:", text)
            if m:
                text = text[m.start():]
        elif label == "diagnosis_block":
            m = re.search(r"Ch\u1ea9n\s*\u0111o\u00e1n\s*:", text)
            if m:
                text = text[m.start():]

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

        preserve_lines = label in {
            "hospital_header",
            "patient_info",
            "diagnosis_block",
            "footer_signature",
        }
        raw_lines = text.splitlines() if preserve_lines else [text.replace("\n", " ")]
        cleaned_lines: list[str] = []

        for raw_line in raw_lines:
            raw_line = re.sub(r"\s+", " ", raw_line).strip()
            cleaned_tokens = []
            for token in raw_line.split(" "):
                token = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹĐđ\-/:.,()%|]", "", token)
                if not token:
                    continue

                letter_count = sum(ch.isalpha() for ch in token)
                digit_count = sum(ch.isdigit() for ch in token)
                punct_count = sum(ch in "-/:.,()%|" for ch in token)
                if letter_count + digit_count == 0:
                    continue
                if punct_count > max(letter_count + digit_count, 1):
                    continue

                cleaned_tokens.append(token)

            line = " ".join(cleaned_tokens).strip()
            line = re.sub(r"([:.,%/\-])\1+", r"\1", line)
            if line:
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines) if preserve_lines else " ".join(cleaned_lines)
        return text.strip()


MedicalReaderHybrid = HybridReader

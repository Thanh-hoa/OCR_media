from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytesseract

from src.utils.image_processing import preprocess_for_ocr


class MedicalReader:
    def __init__(
        self,
        tesseract_cmd: str,
        lang: str = "vie",
        default_psm: int = 6,
        oem: int = 3,
    ) -> None:
        exe_path = Path(tesseract_cmd)
        if not exe_path.exists():
            raise FileNotFoundError(f"Khong tim thay Tesseract tai: {exe_path}")

        pytesseract.pytesseract.tesseract_cmd = str(exe_path)
        self.lang = lang
        self.oem = oem
        self.default_psm = default_psm

    def read_text(self, image_bgr: np.ndarray, label: Optional[str] = None) -> str:
        if image_bgr is None or image_bgr.size == 0:
            return ""

        preprocessed = preprocess_for_ocr(image_bgr, label=label)
        config = self._build_config(label)
        text = pytesseract.image_to_string(preprocessed, lang=self.lang, config=config)
        return self._normalize_text(text)

    def _build_config(self, label: Optional[str]) -> str:
        psm = 6 if label == "test_table" else 7
        return f"--oem {self.oem} --psm {psm} -c preserve_interword_spaces=1"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()


TesseractReader = MedicalReader

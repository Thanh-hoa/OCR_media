from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MESSAGES_DIR = Path(__file__).resolve().parent
_DEFAULT_LOCALE = "vi"
_SUPPORTED_LOCALES = {"vi"}


class MessageTranslator:
    """Tra `message_key` -> câu chữ đã dịch, tương tự messageSource của Spring.

    Nội dung thật nằm trong file `messages_<locale>.json`; muốn thêm ngôn ngữ
    mới thì thêm file catalog và khai báo locale đó trong `_SUPPORTED_LOCALES`.
    """

    def __init__(self) -> None:
        self._catalogs: Dict[str, Dict[str, str]] = {
            locale: self._load_catalog(locale) for locale in _SUPPORTED_LOCALES
        }

    @staticmethod
    def _load_catalog(locale: str) -> Dict[str, str]:
        path = _MESSAGES_DIR / f"messages_{locale}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Không tìm thấy file message cho locale %s: %s", locale, path)
            return {}

    def resolve_locale(self, accept_language: Optional[str]) -> str:
        if accept_language:
            primary = accept_language.split(",")[0].strip().split("-")[0].lower()
            if primary in _SUPPORTED_LOCALES:
                return primary
        return _DEFAULT_LOCALE

    def get_message(self, key: str, locale: str = _DEFAULT_LOCALE, **params: Any) -> str:
        catalog = self._catalogs.get(locale) or self._catalogs.get(_DEFAULT_LOCALE, {})
        template = catalog.get(key, key)
        try:
            return template.format(**params) if params else template
        except (KeyError, IndexError):
            return template


message_translator = MessageTranslator()

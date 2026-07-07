from __future__ import annotations

from typing import Any, Dict, Optional


class AppException(Exception):
    """Exception nghiệp vụ mang message key thay vì câu chữ hardcode.

    Code nghiệp vụ chỉ cần `raise AppException("image.invalid")`; việc dịch
    key thành câu chữ và build response JSON chuẩn do exception handler
    trong `src/api/main.py` xử lý tập trung (xem `MessageTranslator`).
    """

    def __init__(
        self,
        message_key: str,
        status_code: int = 400,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message_key = message_key
        self.status_code = status_code
        self.params = params or {}
        super().__init__(message_key)

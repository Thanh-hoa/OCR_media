from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _strip(v: Optional[str]) -> str:
    return (v or "").strip()


def _normalize_dob(raw: Optional[str]) -> Optional[str]:
    """Chuẩn hóa ngày sinh → yyyy-MM-dd (ISO 8601)."""
    s = _strip(raw)
    if not s:
        return None
    # Bỏ phần giờ nếu có (vd: "15/05/1990 08:00")
    s = re.split(r"\s+\d{1,2}:\d{2}", s)[0].strip()

    for fmt in (
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%Y-%m-%d",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
    ):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y = 1900 + y if y > 30 else 2000 + y
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    logger.warning("Không parse được ngày sinh: %r", s)
    return s 


def _normalize_gender(raw: Optional[str]) -> Optional[str]:
    s = _strip(raw).lower()
    if not s:
        return None
    if "nữ" in s or ("nu" in s and "nam" not in s):
        return "Nữ"
    if "nam" in s:
        return "Nam"
    return "Khác"


def _is_abnormal(value_str: Optional[str], ref_range: Optional[str]) -> bool:
    """
    So sánh value với khoảng tham chiếu.
    Trả True nếu giá trị nằm ngoài khoảng bình thường.
    """
    if not value_str or not ref_range:
        return False

    v_m = re.search(r"(\d+(?:[.,]\d+)?)", value_str.replace(",", "."))
    if not v_m:
        return False
    try:
        value = float(v_m.group(1))
    except ValueError:
        return False

    ref = ref_range.strip()

    m = re.match(
        r"\(?\s*(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*\)?", ref
    )
    if m:
        try:
            lo = float(m.group(1).replace(",", "."))
            hi = float(m.group(2).replace(",", "."))
            return not (lo <= value <= hi)
        except ValueError:
            pass

    # <= max
    m = re.match(r"<=\s*(\d+(?:[.,]\d+)?)", ref)
    if m:
        try:
            return value > float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    # < max
    m = re.match(r"<\s*(\d+(?:[.,]\d+)?)", ref)
    if m:
        try:
            return value >= float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    # >= min
    m = re.match(r">=\s*(\d+(?:[.,]\d+)?)", ref)
    if m:
        try:
            return value < float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    # > min
    m = re.match(r">\s*(\d+(?:[.,]\d+)?)", ref)
    if m:
        try:
            return value <= float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    return False


# ─── MedicalRecordParser ─────────────────────────────────────────────────────

class MedicalRecordParser:
    """
    Biến đổi text OCR thô (đã Gemini-clean) từ 5 vùng thành JSON chuẩn hóa
    sẵn sàng lưu vào PostgreSQL.

    Input  : recordId + dict[region_name → cleaned_text]
    Output : {"recordId": ..., "extractedData": {...}, "labData": [...]}
    """

    _PH_PHONE      = re.compile(
        r"(?:ĐT|DT|Tel(?:ephone)?|SĐT|SDT|Điện\s*thoại|Dien\s*thoai)\s*[:\.]?\s*([\d\s\(\)\-\.\+]+)",
        re.I,
    )
    _PH_ADDR_LABEL = re.compile(r"(?:Địa\s*chỉ|Đ/C)\s*[:\.]?\s*([^\n]+)", re.I)

    # ── Patient patterns ────────────────────────────────────────────────────
    _PP_NAME   = re.compile(r"Họ\s*(?:và\s*)?tên\s*[:\.]?\s*([^\n]+)", re.I)
    _PP_DOB    = re.compile(r"Ngày\s*sinh\s*[:\.]?\s*([^\n]+)", re.I)
    _PP_GENDER = re.compile(r"Giới\s*tính\s*[:\.]?\s*([^\n]+)", re.I)
    _PP_ADDR   = re.compile(r"Địa\s*chỉ\s*[:\.]?\s*([^\n]+)", re.I)
    _PP_BHYT   = re.compile(r"(?:Số\s*thẻ\s*)?BHYT\s*[:\.]?\s*([\w\d]+)", re.I)

    # ── Diagnosis field patterns (label → regex) ────────────────────────────
    _DIAG_FIELDS: List[tuple[str, re.Pattern]] = [
        ("diagnosis",           re.compile(r"Chẩn\s*đoán\s*[:\.]?\s*([^\n]+)", re.I)),
        ("department",          re.compile(r"Khoa\s*(?:/\s*Phòng)?\s*[:\.]?\s*([^\n]+)", re.I)),
        ("facility",            re.compile(r"Cơ\s*sở\s*(?:yêu\s*cầu\s*)?[:\.]?\s*([^\n]+)", re.I)),
        ("sample_collector",    re.compile(r"Người\s*lấy\s*mẫu\s*[:\.]?\s*([^\n]+)", re.I)),
        ("sample_collected_at", re.compile(r"Thời\s*gian\s*lấy\s*mẫu\s*[:\.]?\s*([^\n]+)", re.I)),
        ("sample_receiver",     re.compile(r"Người\s*nhận\s*mẫu\s*[:\.]?\s*([^\n]+)", re.I)),
        ("sample_received_at",  re.compile(r"Thời\s*gian\s*nhận\s*mẫu\s*[:\.]?\s*([^\n]+)", re.I)),
        ("prescribing_doctor",  re.compile(r"(?:BS|Bác\s*sĩ)\s*chỉ\s*định\s*[:\.]?\s*([^\n]+)", re.I)),
        ("specimen_type",       re.compile(r"Loại\s*bệnh\s*phẩm\s*[:\.]?\s*([^\n]+)", re.I)),
        ("sample_status",       re.compile(r"Tình\s*trạng\s*mẫu\s*[:\.]?\s*([^\n]+)", re.I)),
    ]

    # ── Footer title keywords ────────────────────────────────────────────────
    _TITLE_KW = re.compile(
        r"Bác\s*sĩ|Trưởng\s*(?:khoa|phòng)|Phó\s*(?:khoa|phòng|giám\s*đốc)|"
        r"Giám\s*đốc|Kỹ\s*thuật\s*viên|KTV\b|Điều\s*dưỡng|Y\s*tá|Dược\s*sĩ",
        re.I,
    )

    # ── Header rows to skip in test table (no digits in value column) ────────
    _DATE_LINE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def parse(self, ocr_data: Dict[str, Optional[str]]) -> Dict[str, Any]:
        h = self._parse_hospital(_strip(ocr_data.get("hospital_header")))
        p = self._parse_patient(_strip(ocr_data.get("patient_info")))
        d = self._parse_diagnosis(_strip(ocr_data.get("diagnosis_block")))
        lab = self._parse_test_table(_strip(ocr_data.get("test_table")))
        f = self._parse_footer(_strip(ocr_data.get("footer_signature")))

        return {
            "extractedData": {
                "hospital_name":       h.get("name"),
                "hospital_address":    h.get("address"),
                "hospital_phone":      h.get("phone"),
                "patient_name":        p.get("name"),
                "patient_dob":         p.get("dob"),
                "patient_gender":      p.get("gender"),
                "patient_address":     p.get("address"),
                "patient_bhyt":        p.get("bhyt"),
                "diagnosis":           d.get("diagnosis"),
                "department":          d.get("department"),
                "facility":            d.get("facility"),
                "sample_collector":    d.get("sample_collector"),
                "sample_collected_at": d.get("sample_collected_at"),
                "sample_receiver":     d.get("sample_receiver"),
                "sample_received_at":  d.get("sample_received_at"),
                "prescribing_doctor":  d.get("prescribing_doctor"),
                "specimen_type":       d.get("specimen_type"),
                "sample_status":       d.get("sample_status"),
                "signer_title":        f.get("title"),
                "signer_name":         f.get("name"),
            },
            "labData": lab,
        }

    # ─── Hospital Header ─────────────────────────────────────────────────────

    def _parse_hospital(self, text: str) -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {"name": None, "address": None, "phone": None}
        if not text:
            return res

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return res

        # 1. Phone — tìm dòng có nhãn ĐT/Tel/SĐT
        phone_idx = -1
        for i, line in enumerate(lines):
            m = self._PH_PHONE.search(line)
            if m:
                res["phone"] = m.group(1).strip().rstrip(".,;") or None
                phone_idx = i
                break

        # 2. Địa chỉ — ưu tiên nhãn tường minh, rồi suy luận từ vị trí
        addr_m = self._PH_ADDR_LABEL.search(text)
        if addr_m:
            res["address"] = addr_m.group(1).strip() or None
            explicit_addr_line = res["address"]
        else:
            explicit_addr_line = None

        # 3. Tên BV — dòng đầu không phải phone và không phải dòng địa chỉ
        name_idx = -1
        for i, line in enumerate(lines):
            if i == phone_idx:
                continue
            if explicit_addr_line and line == explicit_addr_line:
                continue
            res["name"] = line
            name_idx = i
            break

        # 4. Nếu chưa có địa chỉ, lấy các dòng giữa tên và phone
        if not res.get("address") and name_idx >= 0:
            end = phone_idx if 0 < phone_idx > name_idx else len(lines)
            mid = lines[name_idx + 1 : end]
            if mid:
                res["address"] = " ".join(mid) or None

        return res

    # ─── Patient Info ─────────────────────────────────────────────────────────

    def _parse_patient(self, text: str) -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {
            "name": None, "dob": None, "gender": None,
            "address": None, "bhyt": None,
        }
        if not text:
            return res

        def _get(pat: re.Pattern) -> Optional[str]:
            m = pat.search(text)
            return (m.group(1).strip() or None) if m else None

        raw_name = _get(self._PP_NAME)
        res["name"]    = raw_name.upper() if raw_name else None
        res["dob"]     = _normalize_dob(_get(self._PP_DOB))
        res["gender"]  = _normalize_gender(_get(self._PP_GENDER))
        res["address"] = _get(self._PP_ADDR)
        raw_bhyt = _get(self._PP_BHYT)
        res["bhyt"] = re.sub(r"\s+", "", raw_bhyt) if raw_bhyt else None

        return res

    # ─── Diagnosis Block ─────────────────────────────────────────────────────

    def _parse_diagnosis(self, text: str) -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {k: None for k, _ in self._DIAG_FIELDS}
        if not text:
            return res
        for key, pat in self._DIAG_FIELDS:
            m = pat.search(text)
            val = m.group(1).strip() if m else None
            res[key] = val or None
        return res

    # ─── Test Table ───────────────────────────────────────────────────────────

    def _parse_test_table(self, text: str) -> List[Dict[str, Any]]:
        """
        Mỗi dòng có dạng: TÊN | GIÁ TRỊ | ĐƠN VỊ | THAM CHIẾU
        Bỏ qua dòng header (value không chứa chữ số).
        """
        rows: List[Dict[str, Any]] = []
        if not text:
            return rows

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            name      = parts[0] or None
            val_raw   = parts[1] if len(parts) > 1 else None
            unit      = parts[2] if len(parts) > 2 else None
            ref_range = parts[3] if len(parts) > 3 else None

            # Bỏ dòng thiếu tên hoặc value, bỏ dòng header (value không có số)
            if not name or not val_raw:
                continue
            if not re.search(r"\d", val_raw):
                continue

            val_clean = val_raw
            ref_clean = ref_range.strip() if ref_range else None
            unit_clean = unit.strip() if unit else None

            rows.append({
                "testName":       name,
                "testValue":      val_clean,
                "unit":           unit_clean or None,
                "referenceRange": ref_clean or None,
                "isAbnormal":     _is_abnormal(val_clean, ref_clean),
            })

        return rows

    # ─── Footer / Signature ───────────────────────────────────────────────────

    def _parse_footer(self, text: str) -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {"title": None, "name": None}
        if not text:
            return res

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return res

        # Tìm dòng chứa chức danh
        for i, line in enumerate(lines):
            if self._TITLE_KW.search(line):
                res["title"] = line
                # Tên thường ở dòng kế tiếp, bỏ qua dòng ngày tháng và chức danh khác
                candidates = [
                    l for l in lines[i + 1:]
                    if not self._TITLE_KW.search(l) and not self._DATE_LINE.match(l)
                ]
                res["name"] = candidates[0] if candidates else None
                return res

        # Fallback: không có chức danh nhận dạng được
        res["title"] = lines[0]
        res["name"]  = lines[1] if len(lines) > 1 else None
        return res

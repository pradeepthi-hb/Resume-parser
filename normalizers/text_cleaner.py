import re
from typing import Any, List


_WHITESPACE_RE = re.compile(r"\s+")
_SPLIT_RE = re.compile(r"[\n,;|/]+")
_OCR_SEP_RE = re.compile(r"(?:\|\s*){2,}|[•◦●▪]+|[<>]{2,}|[_=]{3,}")
_JUNK_TOKEN_RE = re.compile(r"^[^A-Za-z0-9]{2,}$")


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_ocr_multiline(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?i)\bend\s*-\s*to\s*-\s*end\b", "end-to-end", text)
    text = re.sub(r"(?i)\bself\s*-\s*service\b", "self-service", text)
    text = re.sub(r"(?i)\b([A-Za-z]+)\s*-\s*\n\s*([A-Za-z]+)\b", r"\1-\2", text)
    text = _OCR_SEP_RE.sub(" ", text)
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"[`~^]+", " ", raw)
        line = re.sub(r"\s*([,;:])\s*", r"\1 ", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip(" -|,;:")
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def clean_ocr_text(value: Any) -> str:
    text = clean_ocr_multiline(value)
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip(" -|,;:")


def split_text_items(value: Any) -> List[str]:
    text = clean_ocr_text(value)
    if not text:
        return []
    items = []
    for raw in _SPLIT_RE.split(text):
        cleaned = clean_ocr_text(raw)
        if cleaned:
            items.append(cleaned)
    return items


def dedupe_case_insensitive(items: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        cleaned = clean_ocr_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def is_junk_fragment(value: Any) -> bool:
    text = clean_ocr_text(value)
    if not text:
        return True
    if _JUNK_TOKEN_RE.match(text):
        return True
    if len(text) <= 1:
        return True
    return False

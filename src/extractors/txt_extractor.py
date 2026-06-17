import os
import re
from typing import Tuple

from src.utils.text import calculate_text_confidence


class TxtExtractor:
    def extract(self, file_path: str) -> Tuple[str, float]:
        if not os.path.exists(file_path):
            return "", 0.0
        with open(file_path, "rb") as handle:
            raw = handle.read()
        text = _decode_text(raw)
        if file_path.lower().endswith(".rtf"):
            text = _strip_rtf(text)
        return text.strip(), calculate_text_confidence(text)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _strip_rtf(text: str) -> str:
    text = re.sub(r"{\\[^{}]+}", " ", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\d* ?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text)

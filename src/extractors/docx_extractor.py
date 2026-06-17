from typing import Tuple

from src.utils.text import extract_text_from_doc, extract_text_from_docx


class DocxExtractor:
    def extract(self, file_path: str) -> Tuple[str, float]:
        return extract_text_from_docx(file_path)


class DocExtractor:
    def extract(self, file_path: str) -> Tuple[str, float]:
        return extract_text_from_doc(file_path)

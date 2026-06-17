import os
from typing import Tuple

from src.extractors.docx_extractor import DocExtractor, DocxExtractor
from src.extractors.image_extractor import ImageExtractor
from src.extractors.pdf_extractor import PdfExtractor
from src.extractors.txt_extractor import TxtExtractor


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".rtf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def get_extractor(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PdfExtractor()
    if ext == ".docx":
        return DocxExtractor()
    if ext == ".doc":
        return DocExtractor()
    if ext in {".txt", ".rtf"}:
        return TxtExtractor()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return ImageExtractor()
    raise ValueError(f"Unsupported resume format: {ext or 'unknown'}")


def extract_text_from_upload(file_path: str) -> Tuple[str, float]:
    return get_extractor(file_path).extract(file_path)

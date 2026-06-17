from typing import Tuple

from src.utils.text import extract_text_from_image


class ImageExtractor:
    def extract(self, file_path: str) -> Tuple[str, float]:
        return extract_text_from_image(file_path)

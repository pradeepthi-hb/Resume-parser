import os
from typing import Any, Callable, Dict, Optional, Tuple

from normalizers.classic_to_resume_schema_v1 import classic_sections_to_resume_schema_v1
from schemas.resume_schema_defaults import RESUME_SCHEMA_VERSION
from services.ai_parser import parse_resume_with_ai
from src.utils.resume_schema_validator import validate_resume_schema_v1


ClassicExtractor = Callable[[str, Optional[str], str], Dict[str, Tuple[str, float]]]


def is_ai_enabled() -> bool:
    return str(os.getenv("AI_ENABLED", "false")).strip().lower() == "true"


def parse_resume_to_schema_v1(
    *,
    text: str,
    filename: str,
    file_path: Optional[str],
    classic_extractor: ClassicExtractor,
) -> Dict[str, Any]:
    mode = _resolve_mode()
    fallback_used = False
    fallback_reason = ""

    if mode == "ai":
        try:
            resume = parse_resume_with_ai(text=text, filename=filename)
            parser_mode = "ai"
        except Exception as exc:
            fallback_used = True
            fallback_reason = str(exc)
            sections = classic_extractor(text, file_path, filename)
            resume = classic_sections_to_resume_schema_v1(sections, raw_text=text)
            parser_mode = "classic"
    else:
        sections = classic_extractor(text, file_path, filename)
        resume = classic_sections_to_resume_schema_v1(sections, raw_text=text)
        parser_mode = "classic"

    validation = validate_resume_schema_v1(resume)
    return {
        "status": "success" if validation["is_valid"] else "error",
        "schema_version": RESUME_SCHEMA_VERSION,
        "parser_mode": parser_mode,
        "ai_enabled": is_ai_enabled(),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "resume": validation["resume"],
        "validation": {
            "is_valid": validation["is_valid"],
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        },
    }


def _resolve_mode() -> str:
    return "ai" if is_ai_enabled() else "classic"

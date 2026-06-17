import re
from typing import Any, Dict, List, Tuple

from schemas.resume_schema_defaults import apply_resume_schema_v1_defaults


def classic_sections_to_resume_schema_v1(
    sections: Dict[str, Tuple[str, float]],
    raw_text: str = "",
) -> Dict[str, Any]:
    get = lambda key: _section_text(sections, key)
    resume = {
        "personal_information": _personal_information(sections, raw_text),
        "summary": get("summary"),
        "work_experience": _text_block_entries(get("experience"), "description"),
        "education": _text_block_entries(get("education"), "description"),
        "skills": {"items": _split_items(get("skills"))},
        "projects": _project_entries(get("projects")),
        "volunteer_experience": _volunteer_entries(get("volunteer")),
        "custom_sections": _custom_sections(sections),
    }
    return apply_resume_schema_v1_defaults(resume)


def _section_text(sections: Dict[str, Tuple[str, float]], key: str) -> str:
    value = sections.get(key, ("", 0.0))
    return str(value[0] or "").strip() if isinstance(value, tuple) else str(value or "").strip()


def _personal_information(sections: Dict[str, Tuple[str, float]], raw_text: str) -> Dict[str, Any]:
    text = raw_text or "\n".join(_section_text(sections, key) for key in sections)
    email = _first_match(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phone = _first_match(r"\+?[0-9][0-9\s().-]{8,}[0-9]", text)
    links = re.findall(r"(?:https?://|www\.)[^\s,;]+", text, flags=re.IGNORECASE)
    personal = {
        "name": _section_text(sections, "name"),
        "email": email,
        "phone": phone,
        "websites": links,
    }
    return {key: value for key, value in personal.items() if value}


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    return match.group(0).strip() if match else ""


def _split_items(text: str) -> List[str]:
    if not text:
        return []
    values = re.split(r"[\n,;•|]+", text)
    return _dedupe([value.strip(" -\t") for value in values if value.strip(" -\t")])


def _text_block_entries(text: str, field_name: str) -> List[Dict[str, str]]:
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]
        blocks = lines if len(lines) > 1 else ([text.strip()] if text.strip() else [])
    return [{field_name: block} for block in blocks]


def _project_entries(text: str) -> List[Dict[str, str]]:
    entries = []
    for item in _text_block_entries(text, "description"):
        description = item["description"]
        first_line = description.splitlines()[0].strip() if description else ""
        entries.append({"title": first_line[:120], "description": description})
    return entries


def _volunteer_entries(text: str) -> List[Dict[str, str]]:
    entries = []
    for item in _text_block_entries(text, "description"):
        description = item["description"]
        first_line = description.splitlines()[0].strip() if description else ""
        entries.append({"organization": first_line, "role": "", "description": description})
    return entries


def _custom_sections(sections: Dict[str, Tuple[str, float]]) -> List[Dict[str, Any]]:
    mapped = {
        "name",
        "email",
        "phone",
        "contact",
        "summary",
        "skills",
        "education",
        "experience",
        "projects",
        "volunteer",
        "links",
        "text",
        "raw_text",
    }
    custom = []
    for key, value in sections.items():
        if key in mapped:
            continue
        text = str(value[0] if isinstance(value, tuple) else value or "").strip()
        if not text:
            continue
        custom.append({"section_title": key.replace("_", " ").title(), "items": _split_items(text) or [text]})
    return custom


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

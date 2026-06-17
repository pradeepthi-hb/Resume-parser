import json
from typing import Any, Dict, List


def resume_schema_v1_section_text(resume: Dict[str, Any], section: str) -> str:
    """Render a single section as plain text for the /extract endpoint.

    This is intentionally tolerant because the UI may store section payloads
    in multiple shapes (the initial “full response” format vs later
    per-section access).

    Supported variants include:
      - ResumeSchemaV1 contract (expected):
          skills: {items:[...]} or skills: {category: [..]}
          education: [{...}, ...]
          work_experience: [{description:...}, ...]
          volunteer_experience: [{description:...}, ...]
      - “new UI” payloads:
          {raw_text: "...", text: "...", value: "...", confidence: ...}
          or lists of such objects.
    """
    if not isinstance(resume, dict):
        return ""

    key = str(section or "").strip().lower().replace("-", "_")

    # If the caller already passes a per-section payload (new UI shape),
    # allow that by returning coerced text.
    if key in resume and isinstance(resume.get(key), (dict, list, str)):
        coerced = _coerce_any_to_text(resume.get(key))
        if coerced:
            return coerced

    personal = resume.get("personal_information") if isinstance(resume.get("personal_information"), dict) else {}

    if key == "name":
        return _text(personal.get("name") or personal.get("full_name"))
    if key == "email":
        return _text(personal.get("email"))
    if key == "phone":
        return _text(personal.get("phone"))
    if key in {"contact", "personal_information"}:
        return _object_lines(personal)
    if key == "summary":
        return _coerce_any_to_text(resume.get("summary"))
    if key == "skills":
        # ResumeSchemaV1: skills is typically a dict; new UI may store raw_text.
        return _skills_lines(resume.get("skills"))
    if key == "education":
        # ResumeSchemaV1: list of objects; new UI: dict {raw_text}.
        return _entries_lines(resume.get("education")) or _coerce_any_to_text(resume.get("education"))
    if key in {"experience", "work_experience"}:
        return _entries_lines(resume.get("work_experience")) or _coerce_any_to_text(resume.get("work_experience"))
    if key == "projects":
        return _entries_lines(resume.get("projects")) or _coerce_any_to_text(resume.get("projects"))
    if key in {"volunteer", "volunteer_experience"}:
        return _entries_lines(resume.get("volunteer_experience")) or _coerce_any_to_text(resume.get("volunteer_experience"))

    custom = _custom_section_text(resume.get("custom_sections"), key)
    if custom:
        return custom

    return _value_lines(resume.get(key))


def _coerce_any_to_text(value: Any) -> str:
    """Best-effort conversion of common “new UI” objects into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()

    # new UI section object: {raw_text|text|value|structured_data}
    if isinstance(value, dict):
        for candidate_key in ("raw_text", "text", "value", "content"):
            v = value.get(candidate_key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # sometimes nested
        sd = value.get("structured_data")
        if isinstance(sd, dict):
            for candidate_key in ("raw_text", "text", "value", "content", "description"):
                v = sd.get(candidate_key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # or {items:[...]}
            if "items" in sd:
                return _coerce_any_to_text(sd.get("items"))

        # arrays inside dict
        if "items" in value:
            return _coerce_any_to_text(value.get("items"))

        return _value_lines(value)

    # list of things
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            t = _coerce_any_to_text(item)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()

    return str(value).strip()



def _custom_section_text(custom_sections: Any, key: str) -> str:
    if not isinstance(custom_sections, list):
        return ""
    wanted = key.replace("_", " ").lower()
    aliases = {
        "awards": {"awards", "achievements", "awards achievements"},
        "certifications": {"certifications", "certification", "licenses", "courses"},
        "languages": {"languages", "language skills", "known languages"},
        "references": {"references", "referees"},
        "interests": {"interests", "hobbies", "passions"},
        "publications": {"publications", "publication"},
    }.get(key, {wanted})

    lines: List[str] = []
    for section in custom_sections:
        if not isinstance(section, dict):
            continue
        title = _text(section.get("section_title")).lower()
        if title not in aliases:
            continue
        lines.extend(_list_items(section.get("items")))
    return "\n".join(lines)


def _skills_lines(skills: Any) -> str:
    if isinstance(skills, dict):
        lines: List[str] = []
        for value in skills.values():
            lines.extend(_list_items(value))
        return "\n".join(dict.fromkeys(line for line in lines if line))
    return _value_lines(skills)


def _entries_lines(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    entries = []
    for item in value:
        rendered = _value_lines(item)
        if rendered:
            entries.append(rendered)
    return "\n\n".join(entries)


def _value_lines(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_list_items(value))
    if isinstance(value, dict):
        return _object_lines(value)
    return str(value).strip()


def _object_lines(value: Dict[str, Any]) -> str:
    lines = []
    for key, item in value.items():
        rendered = _value_lines(item)
        if rendered:
            label = str(key).replace("_", " ").title()
            lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def _list_items(value: Any) -> List[str]:
    if isinstance(value, list):
        items = []
        for item in value:
            text = _value_lines(item)
            if text:
                items.append(text)
        return items
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    if isinstance(value, dict):
        text = _object_lines(value)
        return [text] if text else []
    if value:
        return [str(value).strip()]
    return []


def schema_preview_html(resume: Dict[str, Any]) -> str:
    """Small server-safe fallback renderer for tests or non-JS callers."""
    return json.dumps(resume or {}, indent=2, ensure_ascii=True)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

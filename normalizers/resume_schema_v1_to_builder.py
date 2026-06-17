from typing import Any, Dict, List


class ResumeSchemaV1ToBuilderMapper:
    """Temporary adapter from ResumeSchemaV1 to the Resume Builder payload.

    This module intentionally has no parser, Flask, AI, or validation
    dependencies so it can be moved into the Resume Builder project later.
    """

    def map(self, resume: Dict[str, Any], local_resume_id: int = 1, file_type: str = "pdf") -> Dict[str, Any]:
        personal = _dict(resume.get("personal_information"))
        return {
            "personal": {
                "firstName": _first_name(personal),
                "lastName": _last_name(personal),
                "profession": _text(personal.get("profession") or personal.get("job_title") or personal.get("title")),
                "onet_code": _text(personal.get("onet_code")),
                "city": _text(personal.get("city")),
                "country": _text(personal.get("country")),
                "pincode": _text(personal.get("pincode") or personal.get("postal_code")),
                "phone": _text(personal.get("phone")),
                "email": _text(personal.get("email")),
                "linkedin": _text(personal.get("linkedin")),
                "websites": _string_list(personal.get("websites") or personal.get("website")),
            },
            "summary": _text(resume.get("summary")),
            "skills": _flatten_skills(resume.get("skills")),
            "education": [_education(item) for item in _object_list(resume.get("education"))],
            "experience": [_experience(item) for item in _object_list(resume.get("work_experience"))],
            "projects": [_project(item) for item in _object_list(resume.get("projects"))],
            "certifications": _custom_named_items(resume, {"certifications", "certification", "licenses", "courses"}),
            "languages": _custom_named_items(resume, {"languages", "language skills", "known languages"}),
            "template": "classic",
            "_meta": {"local_resume_id": int(local_resume_id or 1)},
            "file_type": _text(file_type).lower() or "pdf",
        }


def get_resume_schema_v1_to_builder_mapper() -> ResumeSchemaV1ToBuilderMapper:
    return ResumeSchemaV1ToBuilderMapper()


def build_resume_builder_response(
    resume: Dict[str, Any],
    validation: Dict[str, Any],
    local_resume_id: int = 1,
    file_type: str = "pdf",
) -> Dict[str, Any]:
    mapper = get_resume_schema_v1_to_builder_mapper()
    is_valid = bool(validation.get("is_valid", True)) if isinstance(validation, dict) else True
    return {
        "success": is_valid,
        "confidence": 1.0 if is_valid else 0.0,
        "resume_data": mapper.map(resume, local_resume_id=local_resume_id, file_type=file_type),
        "raw_parser_output": resume,
        "raw_resume_schema_v1": resume,
        "parser_metadata": {
            "mapping_version": "resume-schema-v1-to-builder-v1",
            "source_schema": "resume_schema_v1",
            "validation": validation or {"is_valid": is_valid, "errors": [], "warnings": []},
            "normalization_logs": {
                "builder_compatibility_issues": [] if is_valid else (validation or {}).get("errors", []),
                "unmapped_fields": [],
                "normalization_corrections": [],
                "invalid_dates": [],
                "low_confidence_mappings": [],
                "unsupported_structures": [],
            },
        },
    }


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    if value:
        return [_text(value)]
    return []


def _object_list(value: Any) -> List[Dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _first_name(personal: Dict[str, Any]) -> str:
    explicit = _text(personal.get("firstName") or personal.get("first_name"))
    if explicit:
        return explicit
    parts = _text(personal.get("name") or personal.get("full_name")).split()
    return parts[0] if parts else ""


def _last_name(personal: Dict[str, Any]) -> str:
    explicit = _text(personal.get("lastName") or personal.get("last_name"))
    if explicit:
        return explicit
    parts = _text(personal.get("name") or personal.get("full_name")).split()
    return " ".join(parts[1:]) if len(parts) > 1 else ""


def _flatten_skills(skills: Any) -> List[str]:
    if isinstance(skills, list):
        return list(dict.fromkeys(_text(item) for item in skills if _text(item)))
    if not isinstance(skills, dict):
        return []
    flattened: List[str] = []
    for value in skills.values():
        if isinstance(value, list):
            flattened.extend(_text(item) for item in value if _text(item))
        elif value:
            flattened.append(_text(value))
    return list(dict.fromkeys(flattened))


def _education(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "school": _text(item.get("school") or item.get("institution") or item.get("university")),
        "location": _text(item.get("location")or item.get("place")),
        "degree": _text(item.get("degree")),
        "field": _text(item.get("field") or item.get("field_of_study")),
        "gradMonth": _text(item.get("gradMonth") or item.get("graduation_month")),
        "gradYear": _text(item.get("gradYear") or item.get("graduation_year") or item.get("end_date")),
        "gpa": _text(item.get("gpa") or item.get("grade")or item.get("percentage")or item.get("marks")),
    }


def _experience(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": _text(item.get("title") or item.get("job_title") or item.get("role")),
        "employer": _text(item.get("employer") or item.get("company") or item.get("organization")),
        "location": _text(item.get("location")),
        "description": _description(item),
        "startMonth": _text(item.get("startMonth") or item.get("start_month")),
        "startYear": _text(item.get("startYear") or item.get("start_year") or item.get("start_date")),
        "endMonth": _text(item.get("endMonth") or item.get("end_month")),
        "endYear": _text(item.get("endYear") or item.get("end_year") or item.get("end_date")),
        "current": bool(item.get("current", False)),
        "onet_code": _text(item.get("onet_code")),
    }


def _project(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": _text(item.get("title") or item.get("name")),
        "description": _description(item),
        "role": _text(item.get("role")),
        "tools": ", ".join(_string_list(item.get("technologies") or item.get("tools"))),
        "url": _text(item.get("url") or item.get("link")),
    }


def _description(item: Dict[str, Any]) -> str:
    value = item.get("description")
    if isinstance(value, list):
        return "\n".join(_text(part) for part in value if _text(part))
    responsibilities = item.get("responsibilities")
    if isinstance(responsibilities, list):
        return "\n".join(_text(part) for part in responsibilities if _text(part))
    return _text(value)


def _custom_named_items(resume: Dict[str, Any], names: set) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for section in _object_list(resume.get("custom_sections")):
        title = _text(section.get("section_title")).lower()
        if title not in names:
            continue
        for item in _string_list(section.get("items")):
            output.append({"name": item, "issuingOrg": "", "achievedDate": "", "url": ""})
    return output

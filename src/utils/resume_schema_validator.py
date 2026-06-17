from typing import Any, Dict, List

from schemas.resume_schema_defaults import apply_resume_schema_v1_defaults


REQUIRED_TYPES = {
    "personal_information": dict,
    "work_experience": list,
    "education": list,
    "skills": dict,
}

OPTIONAL_TYPES = {
    "summary": str,
    "projects": list,
    "volunteer_experience": list,
    "custom_sections": list,
}


def validate_resume_schema_v1(value: Any) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(value, dict):
        return {
            "is_valid": False,
            "errors": ["resume must be an object"],
            "warnings": [],
            "resume": apply_resume_schema_v1_defaults({}),
        }

    resume = apply_resume_schema_v1_defaults(value)

    for key, expected_type in REQUIRED_TYPES.items():
        if key not in value:
            warnings.append(f"missing required key '{key}' was defaulted")
        if not isinstance(resume.get(key), expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}")

    for key, expected_type in OPTIONAL_TYPES.items():
        if key not in value:
            warnings.append(f"missing optional key '{key}' was defaulted")
        if not isinstance(resume.get(key), expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}")

    _validate_array_objects(resume, "work_experience", errors)
    _validate_array_objects(resume, "education", errors)
    _validate_array_objects(resume, "projects", errors)
    _validate_array_objects(resume, "volunteer_experience", errors)
    _validate_custom_sections(resume, errors)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "resume": resume,
    }


def _validate_array_objects(resume: Dict[str, Any], key: str, errors: List[str]) -> None:
    for index, item in enumerate(resume.get(key, [])):
        if not isinstance(item, dict):
            errors.append(f"'{key}[{index}]' must be object")


def _validate_custom_sections(resume: Dict[str, Any], errors: List[str]) -> None:
    for index, item in enumerate(resume.get("custom_sections", [])):
        if not isinstance(item, dict):
            errors.append(f"'custom_sections[{index}]' must be object")
            continue
        if not isinstance(item.get("section_title", ""), str):
            errors.append(f"'custom_sections[{index}].section_title' must be string")
        if not isinstance(item.get("items", []), list):
            errors.append(f"'custom_sections[{index}].items' must be array")

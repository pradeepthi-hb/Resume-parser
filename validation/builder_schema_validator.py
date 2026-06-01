import re
from typing import Any, Dict, List, Set


TOP_LEVEL_KEYS: Set[str] = {
    "personal",
    "summary",
    "skills",
    "education",
    "experience",
    "projects",
    "certifications",
    "languages",
    "template",
    "_meta",
    "file_type",
}

PERSONAL_KEYS: Set[str] = {
    "firstName",
    "lastName",
    "profession",
    "onet_code",
    "city",
    "country",
    "pincode",
    "phone",
    "email",
    "linkedin",
    "websites",
}

EDUCATION_KEYS: Set[str] = {
    "school",
    "location",
    "field",
    "degree",
    "gradMonth",
    "gradYear",
    "gpa",
}

EXPERIENCE_KEYS: Set[str] = {
    "title",
    "employer",
    "location",
    "description",
    "startMonth",
    "startYear",
    "endMonth",
    "endYear",
    "current",
    "onet_code",
    "originalDescription",
    "lastTemplateIndex",
}

PROJECT_KEYS: Set[str] = {"title", "description", "role", "tools", "url"}
CERT_KEYS: Set[str] = {"name", "issuingOrg", "achievedDate", "url"}
LANGUAGE_KEYS: Set[str] = {"name", "level"}

LANGUAGE_LEVEL_ENUM = {"Beginner", "Intermediate", "Advanced", "Fluent", "Native"}
MONTH_ENUM = {
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
YEAR_RE = re.compile(r"^(?:|19\d{2}|20\d{2})$")
ACHIEVED_DATE_RE = re.compile(
    r"^(?:|19\d{2}|20\d{2}|"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19\d{2}|20\d{2}))$"
)


def _validate_exact_keys(
    value: Any,
    expected_keys: Set[str],
    path: str,
    errors: List[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return
    missing = sorted(expected_keys - set(value.keys()))
    extra = sorted(set(value.keys()) - expected_keys)
    if missing:
        errors.append(f"{path} missing keys: {missing}")
    if extra:
        errors.append(f"{path} has unsupported keys: {extra}")


def _validate_string(value: Any, path: str, errors: List[str]) -> None:
    if not isinstance(value, str):
        errors.append(f"{path} must be a string")


def _validate_string_list(value: Any, path: str, errors: List[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} must be an array")
        return
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{path}[{idx}] must be a string")


def _validate_year(value: Any, path: str, errors: List[str]) -> None:
    if not isinstance(value, str) or not YEAR_RE.match(value):
        errors.append(f"{path} must be '' or a four-digit year")


def _validate_month(value: Any, path: str, errors: List[str]) -> None:
    if not isinstance(value, str) or value not in MONTH_ENUM:
        errors.append(f"{path} must be one of the canonical month values or ''")


def validate_builder_schema(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    _validate_exact_keys(resume_data, TOP_LEVEL_KEYS, "root", errors)
    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    _validate_exact_keys(resume_data["personal"], PERSONAL_KEYS, "personal", errors)
    for key in sorted(PERSONAL_KEYS - {"websites"}):
        _validate_string(resume_data["personal"].get(key), f"personal.{key}", errors)
    _validate_string_list(resume_data["personal"].get("websites"), "personal.websites", errors)

    _validate_string(resume_data.get("summary"), "summary", errors)
    _validate_string_list(resume_data.get("skills"), "skills", errors)

    if not isinstance(resume_data.get("education"), list):
        errors.append("education must be an array")
    else:
        for idx, item in enumerate(resume_data["education"]):
            _validate_exact_keys(item, EDUCATION_KEYS, f"education[{idx}]", errors)
            for key in ["school", "location", "field", "degree", "gpa"]:
                _validate_string(item.get(key), f"education[{idx}].{key}", errors)
            _validate_month(item.get("gradMonth"), f"education[{idx}].gradMonth", errors)
            _validate_year(item.get("gradYear"), f"education[{idx}].gradYear", errors)

    if not isinstance(resume_data.get("experience"), list):
        errors.append("experience must be an array")
    else:
        for idx, item in enumerate(resume_data["experience"]):
            _validate_exact_keys(item, EXPERIENCE_KEYS, f"experience[{idx}]", errors)
            for key in [
                "title",
                "employer",
                "location",
                "description",
                "startMonth",
                "startYear",
                "endMonth",
                "endYear",
                "onet_code",
                "originalDescription",
            ]:
                _validate_string(item.get(key), f"experience[{idx}].{key}", errors)
            _validate_month(item.get("startMonth"), f"experience[{idx}].startMonth", errors)
            _validate_month(item.get("endMonth"), f"experience[{idx}].endMonth", errors)
            _validate_year(item.get("startYear"), f"experience[{idx}].startYear", errors)
            _validate_year(item.get("endYear"), f"experience[{idx}].endYear", errors)
            if not isinstance(item.get("current"), bool):
                errors.append(f"experience[{idx}].current must be boolean")
            if not isinstance(item.get("lastTemplateIndex"), int):
                errors.append(f"experience[{idx}].lastTemplateIndex must be integer")

    if not isinstance(resume_data.get("projects"), list):
        errors.append("projects must be an array")
    else:
        for idx, item in enumerate(resume_data["projects"]):
            _validate_exact_keys(item, PROJECT_KEYS, f"projects[{idx}]", errors)
            for key in sorted(PROJECT_KEYS):
                _validate_string(item.get(key), f"projects[{idx}].{key}", errors)

    if not isinstance(resume_data.get("certifications"), list):
        errors.append("certifications must be an array")
    else:
        for idx, item in enumerate(resume_data["certifications"]):
            _validate_exact_keys(item, CERT_KEYS, f"certifications[{idx}]", errors)
            for key in ["name", "issuingOrg", "url"]:
                _validate_string(item.get(key), f"certifications[{idx}].{key}", errors)
            achieved_date = item.get("achievedDate")
            if not isinstance(achieved_date, str) or not ACHIEVED_DATE_RE.match(achieved_date):
                errors.append(
                    f"certifications[{idx}].achievedDate must be '', 'YYYY', or 'Month YYYY'"
                )

    if not isinstance(resume_data.get("languages"), list):
        errors.append("languages must be an array")
    else:
        for idx, item in enumerate(resume_data["languages"]):
            _validate_exact_keys(item, LANGUAGE_KEYS, f"languages[{idx}]", errors)
            _validate_string(item.get("name"), f"languages[{idx}].name", errors)
            level = item.get("level")
            if not isinstance(level, str):
                errors.append(f"languages[{idx}].level must be a string")
            elif level not in LANGUAGE_LEVEL_ENUM:
                errors.append(
                    f"languages[{idx}].level must be one of {sorted(LANGUAGE_LEVEL_ENUM)}"
                )

    _validate_string(resume_data.get("template"), "template", errors)
    if not isinstance(resume_data.get("_meta"), dict):
        errors.append("_meta must be an object")
    else:
        if set(resume_data["_meta"].keys()) != {"local_resume_id"}:
            errors.append("_meta must only contain local_resume_id")
        if not isinstance(resume_data["_meta"].get("local_resume_id"), int):
            errors.append("_meta.local_resume_id must be integer")
    _validate_string(resume_data.get("file_type"), "file_type", errors)

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}


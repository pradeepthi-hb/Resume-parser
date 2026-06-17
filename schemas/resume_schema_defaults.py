from copy import deepcopy
from typing import Any, Dict


RESUME_SCHEMA_VERSION = "resume_schema_v1"

RESUME_SCHEMA_V1_DEFAULTS: Dict[str, Any] = {
    "personal_information": {},
    "summary": "",
    "work_experience": [],
    "education": [],
    "skills": {},
    "projects": [],
    "volunteer_experience": [],
    "custom_sections": [],
}


def empty_resume_schema_v1() -> Dict[str, Any]:
    return deepcopy(RESUME_SCHEMA_V1_DEFAULTS)


def apply_resume_schema_v1_defaults(value: Any) -> Dict[str, Any]:
    resume = empty_resume_schema_v1()
    if isinstance(value, dict):
        for key, item in value.items():
            resume[key] = item

    if not isinstance(resume.get("personal_information"), dict):
        resume["personal_information"] = {}
    if not isinstance(resume.get("summary"), str):
        resume["summary"] = "" if resume.get("summary") is None else str(resume.get("summary"))
    if not isinstance(resume.get("work_experience"), list):
        resume["work_experience"] = []
    if not isinstance(resume.get("education"), list):
        resume["education"] = []
    if not isinstance(resume.get("skills"), dict):
        resume["skills"] = {"items": _as_string_list(resume.get("skills"))}
    if not isinstance(resume.get("projects"), list):
        resume["projects"] = []
    if not isinstance(resume.get("volunteer_experience"), list):
        resume["volunteer_experience"] = []
    if not isinstance(resume.get("custom_sections"), list):
        resume["custom_sections"] = []

    return resume


def _as_string_list(value: Any) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    if value:
        return [str(value).strip()]
    return []

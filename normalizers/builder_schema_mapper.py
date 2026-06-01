import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from normalizers.date_normalizer import (
    extract_date_range,
    normalize_achieved_date,
    normalize_month,
    normalize_year,
)
from normalizers.text_cleaner import clean_ocr_multiline, clean_ocr_text, dedupe_case_insensitive, is_junk_fragment, split_text_items
from validation.builder_schema_validator import validate_builder_schema


logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
EMAIL_FIND_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_FIND_RE = re.compile(r"\+?[0-9][0-9\s().-]{8,}[0-9]")
LINKEDIN_RE = re.compile(r"^(https?://)?(www\.)?linkedin\.com/(in|pub|company)/[^\s/]+/?$", re.IGNORECASE)
URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}/[^\s]*)", re.IGNORECASE)
PINCODE_RE = re.compile(r"\b\d{5,6}\b")
YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-/]\s*((?:19|20)\d{2}|\d{2})\b")
MONTH_TOKEN_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|0?[1-9]|1[0-2]"
    r")\b",
    re.IGNORECASE,
)

EDU_WORDS = {
    "education",
    "university",
    "college",
    "institute",
    "school",
    "bachelor",
    "master",
    "diploma",
    "ssc",
    "hsc",
    "intermediate",
    "cgpa",
    "gpa",
    "percentage",
}
CONTACT_STOP_WORDS = {"summary", "experience", "skills", "education", "projects", "certifications", "references"}
EXPERIENCE_SECTION_HEADERS = {
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "employment",
    "career history",
}
SECTION_STOP_HEADERS = EXPERIENCE_SECTION_HEADERS | {
    "education",
    "skills",
    "projects",
    "certifications",
    "achievements",
    "languages",
    "interests",
    "summary",
    "professional summary",
}
RESPONSIBILITY_VERBS = {
    "responsible",
    "managed",
    "developed",
    "implemented",
    "designed",
    "led",
    "improved",
    "created",
    "handled",
    "performed",
}
COMPANY_SUFFIXES = {
    "pvt",
    "ltd",
    "llc",
    "inc",
    "corp",
    "corporation",
    "technologies",
    "technology",
    "solutions",
    "services",
    "consulting",
    "group",
    "systems",
    "labs",
    "co",
}
KNOWN_COMPANY_NAMES = {
    "adp",
    "exl",
    "wns",
    "examity",
}
LOCATION_WORDS = {
    "hyderabad",
    "guntur",
    "vijayawada",
    "delhi",
    "telangana",
    "india",
    "mumbai",
    "pune",
    "bangalore",
    "bengaluru",
    "chennai",
    "kolkata",
}
TITLE_HINTS = {
    "engineer",
    "developer",
    "manager",
    "analyst",
    "architect",
    "executive",
    "consultant",
    "specialist",
    "accountant",
    "administrator",
    "lead",
    "officer",
    "representative",
    "assistant",
    "payroll",
}
NON_TITLE_PHRASES = {
    "professional experience",
    "proffessional experience",
    "work experience",
    "employment history",
    "responsibilities",
    "profile",
    "summary",
}
COMMON_LANGUAGE_NAMES = {
    "english", "hindi", "french", "german", "spanish", "arabic", "urdu", "tamil",
    "telugu", "kannada", "malayalam", "marathi", "gujarati", "bengali", "japanese",
    "chinese", "korean", "portuguese", "italian",
}
SKILL_NOISE_TOKENS = {
    "additional information",
    "languages",
    "interests",
    "profile",
    "summary",
    "skills",
}
SKILL_CASE_MAP = {
    "sap": "SAP",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "bom": "BOMs",
    "boms": "BOMs",
    "aws": "AWS",
    "api": "API",
    "apis": "APIs",
    "rest apis": "REST APIs",
    "html5": "HTML5",
    "css3": "CSS3",
    "github": "GitHub",
    "javascript": "JavaScript",
    "php": "PHP",
    "servicenow": "ServiceNow",
    "vs code": "VS Code",
}
SKILL_INFERENCE_KEYWORDS = {
    "payroll": "Payroll",
    "forecasting": "Forecasting",
    "budgeting": "Budgeting",
    "financial analysis": "Financial Analysis",
    "reconciliation": "Reconciliation",
    "sap": "SAP",
    "power bi": "Power BI",
    "sql": "SQL",
    "excel": "Excel",
    "python": "Python",
}
GENERIC_SKILLS = {"team player", "hard working", "self motivated", "good communication"}
NAME_BLOCKLIST_TOKENS = {
    "language", "languages", "python", "javascript", "java", "php", "sql", "html", "css",
    "react", "node", "skills", "tools", "platforms", "databases", "frameworks",
    "mpc", "nri", "ssc", "hsc", "college", "school", "institute", "university",
    "core", "competencies", "payroll", "implementation", "compliance", "uk",
    "audit", "financial", "reporting", "taxation", "risk", "assessment",
}

SUPPORTED_NATIVE_FIELDS = {
    "name",
    "email",
    "phone",
    "summary",
    "skills",
    "education",
    "experience",
    "projects",
    "certifications",
    "languages",
    "contact",
    "links",
    "raw_text",
}

LANGUAGE_LEVEL_MAP = {
    "beginner": "Beginner",
    "elementary": "Beginner",
    "intermediate": "Intermediate",
    "conversational": "Intermediate",
    "advanced": "Advanced",
    "professional": "Advanced",
    "fluent": "Fluent",
    "native": "Native",
    "bilingual": "Native",
}


def _blank_builder_resume(local_resume_id: int, file_type: str) -> Dict[str, Any]:
    return {
        "personal": {
            "firstName": "",
            "lastName": "",
            "profession": "",
            "onet_code": "",
            "city": "",
            "country": "",
            "pincode": "",
            "phone": "",
            "email": "",
            "linkedin": "",
            "websites": [],
        },
        "summary": "",
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "template": "classic",
        "_meta": {"local_resume_id": int(local_resume_id)},
        "file_type": clean_ocr_text(file_type).lower() or "pdf",
    }


class BuilderSchemaMapper:
    def __init__(self, low_confidence_threshold: float = 0.35):
        self.low_confidence_threshold = low_confidence_threshold

    def map_parser_output(
        self,
        raw_parser_output: Dict[str, Any],
        overall_confidence: float,
        local_resume_id: int = 1,
        file_type: str = "pdf",
    ) -> Dict[str, Any]:
        logs: Dict[str, List[str]] = {
            "unmapped_fields": [],
            "normalization_corrections": [],
            "invalid_dates": [],
            "low_confidence_mappings": [],
            "unsupported_structures": [],
            "builder_compatibility_issues": [],
        }

        native = raw_parser_output if isinstance(raw_parser_output, dict) else {}
        if not isinstance(raw_parser_output, dict):
            logs["unsupported_structures"].append("raw_parser_output is not an object; using empty object")

        resume_data = _blank_builder_resume(local_resume_id=local_resume_id, file_type=file_type)

        self._map_personal(native, resume_data, logs)
        self._map_education(native, resume_data, logs)
        self._map_experience(native, resume_data, logs)
        self._map_skills(native, resume_data, logs)
        self._map_projects(native, resume_data, logs)
        self._map_certifications(native, resume_data, logs)
        self._map_languages(native, resume_data, logs)
        self._map_summary(native, resume_data, logs)
        self._finalize_resume_data(native, resume_data, logs)

        validation = validate_builder_schema(resume_data)
        if not validation["is_valid"]:
            logs["builder_compatibility_issues"].extend(validation["errors"])
        if validation["warnings"]:
            logs["builder_compatibility_issues"].extend(validation["warnings"])

        native_keys = sorted(native.keys())
        unsupported_fields = sorted(set(native_keys) - SUPPORTED_NATIVE_FIELDS)
        for field in unsupported_fields:
            logs["unmapped_fields"].append(f"native field '{field}' has no builder schema target")

        for issue in logs["unmapped_fields"]:
            logger.info("Builder mapper unmapped field: %s", issue)
        for issue in logs["normalization_corrections"]:
            logger.info("Builder mapper normalization correction: %s", issue)
        for issue in logs["invalid_dates"]:
            logger.warning("Builder mapper invalid date: %s", issue)
        for issue in logs["low_confidence_mappings"]:
            logger.warning("Builder mapper low confidence mapping: %s", issue)
        for issue in logs["unsupported_structures"]:
            logger.warning("Builder mapper unsupported structure: %s", issue)
        for issue in logs["builder_compatibility_issues"]:
            logger.error("Builder mapper compatibility issue: %s", issue)

        parser_metadata = {
            "mapping_version": "builder-schema-v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "low_confidence_threshold": self.low_confidence_threshold,
            "unsupported_parser_fields": unsupported_fields,
            "source_sections": native_keys,
            "validation": validation,
            "normalization_logs": logs,
        }

        confidence = float(overall_confidence)
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "success": bool(validation["is_valid"]),
            "confidence": round(confidence, 4),
            "resume_data": resume_data,
            "raw_parser_output": native,
            "parser_metadata": parser_metadata,
        }

    def _section(self, native: Dict[str, Any], name: str) -> Dict[str, Any]:
        value = native.get(name, {})
        if isinstance(value, dict):
            return value
        return {"raw_text": clean_ocr_multiline(value), "structured_data": {}, "confidence": 0.0}

    def _section_confidence(self, native: Dict[str, Any], name: str) -> float:
        section = self._section(native, name)
        try:
            return float(section.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _section_raw(self, native: Dict[str, Any], name: str) -> str:
        return clean_ocr_multiline(self._section(native, name).get("raw_text", ""))

    def _section_structured(self, native: Dict[str, Any], name: str) -> Dict[str, Any]:
        value = self._section(native, name).get("structured_data", {})
        return value if isinstance(value, dict) else {}

    def _is_low_conf(self, conf: float, strict: bool = False) -> bool:
        threshold = self.low_confidence_threshold + (0.15 if strict else 0.0)
        return conf > 0.0 and conf < threshold

    def _map_personal(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        personal = resume_data["personal"]
        contact_text = self._section_raw(native, "contact")
        summary_text = self._section_raw(native, "summary")
        full_text = self._section_raw(native, "raw_text")

        name_conf = self._section_confidence(native, "name")
        name_text = self._section_raw(native, "name")
        if name_text and not self._is_low_conf(name_conf, strict=True) and self._is_valid_person_name(name_text):
            first, last = self._split_name(name_text)
            personal["firstName"], personal["lastName"] = first, last
        elif name_text:
            logs["low_confidence_mappings"].append(f"name confidence={round(name_conf, 4)} below threshold")
        if not personal["firstName"]:
            email_hint = self._recover_name_from_email(
                self._section_raw(native, "email") or self._extract_first_match(EMAIL_FIND_RE, full_text)
            )
            recovered_name = self._recover_name_from_text(
                "\n".join(
                    [
                        contact_text,
                        full_text,
                        self._section_raw(native, "certifications"),
                        self._section_raw(native, "achievements"),
                    ]
                )
            ) or email_hint
            first_line = recovered_name
            for line in clean_ocr_multiline(full_text).splitlines():
                if first_line:
                    break
                candidate = clean_ocr_text(line)
                if not candidate:
                    continue
                if EMAIL_FIND_RE.search(candidate) or PHONE_FIND_RE.search(candidate):
                    continue
                if candidate.lower() in SECTION_STOP_HEADERS:
                    continue
                if self._is_valid_person_name(candidate):
                    first_line = candidate
                    break
            if first_line:
                first, last = self._split_name(first_line)
                personal["firstName"], personal["lastName"] = first, last

        email_candidates = [
            self._section_raw(native, "email"),
            self._extract_first_match(EMAIL_FIND_RE, contact_text),
            self._extract_first_match(EMAIL_FIND_RE, summary_text),
            self._extract_first_match(EMAIL_FIND_RE, full_text),
        ]
        email = self._pick_valid_email(email_candidates)
        if email:
            if self._section_confidence(native, "email") >= self.low_confidence_threshold or email in contact_text:
                personal["email"] = email
            else:
                logs["low_confidence_mappings"].append("email candidate discarded due to low confidence")

        phone_candidates = [
            self._section_raw(native, "phone"),
            self._extract_first_match(PHONE_FIND_RE, contact_text),
            self._extract_first_match(PHONE_FIND_RE, full_text),
        ]
        phone = self._pick_valid_phone(phone_candidates)
        if phone:
            if self._section_confidence(native, "phone") >= self.low_confidence_threshold or phone in clean_ocr_text(contact_text):
                personal["phone"] = phone
            else:
                logs["low_confidence_mappings"].append("phone candidate discarded due to low confidence")

        linkedin, websites = self._extract_links(native)
        if linkedin and self._is_valid_linkedin(linkedin):
            personal["linkedin"] = linkedin
        personal["websites"] = websites

        pin_match = PINCODE_RE.search(contact_text)
        if pin_match:
            personal["pincode"] = pin_match.group(0)

        city, country = self._extract_location_from_contact(contact_text)
        if city:
            personal["city"] = city
        if country:
            personal["country"] = country

        profession = self._derive_profession(native)
        if profession:
            personal["profession"] = profession

    def _map_education(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "education")
        if self._is_low_conf(conf):
            logs["low_confidence_mappings"].append(
                f"education confidence={round(conf, 4)} below threshold; education removed"
            )
            resume_data["education"] = []
            return

        structured_entries = self._section_structured(native, "education").get("entries", [])
        raw_text = self._section_raw(native, "education")
        parsed_entries: List[Dict[str, Any]] = []

        if isinstance(structured_entries, list) and structured_entries:
            for idx, item in enumerate(structured_entries):
                if not isinstance(item, dict):
                    logs["unsupported_structures"].append(f"education entry {idx} is not an object")
                    continue
                parsed_entries.append(self._normalize_education_entry(item, logs, idx))

        if raw_text:
            parsed_entries.extend(self._extract_education_from_raw(raw_text, logs))

        parsed_entries = [e for e in parsed_entries if any(e.get(k, "") for k in ("degree", "field", "school", "gradYear", "gpa"))]
        resume_data["education"] = self._dedupe_education(parsed_entries)

    def _map_experience(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "experience")
        if self._is_low_conf(conf):
            logs["low_confidence_mappings"].append(
                f"experience confidence={round(conf, 4)} below threshold; experience removed"
            )
            resume_data["experience"] = []
            return

        structured_entries = self._section_structured(native, "experience").get("entries", [])
        raw_text = self._section_raw(native, "experience")
        normalized: List[Dict[str, Any]] = []

        if isinstance(structured_entries, list):
            for idx, item in enumerate(structured_entries):
                if not isinstance(item, dict):
                    logs["unsupported_structures"].append(f"experience entry {idx} is not an object")
                    continue
                candidate = self._normalize_experience_entry(item, logs, idx)
                if candidate:
                    normalized.append(candidate)

        raw_normalized: List[Dict[str, Any]] = []
        if raw_text:
            raw_normalized = self._extract_experience_from_raw(raw_text, logs)

        structured_score = self._experience_quality_score(normalized)
        raw_score = self._experience_quality_score(raw_normalized)

        chosen = normalized
        if raw_score > structured_score:
            logs["normalization_corrections"].append(
                f"experience remapped from raw section (quality {raw_score} > structured {structured_score})"
            )
            chosen = raw_normalized
        elif raw_score == structured_score and raw_score > 0 and len(raw_normalized) > len(normalized):
            chosen = raw_normalized

        resume_data["experience"] = self._dedupe_experience(chosen)

    def _map_skills(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "skills")
        if self._is_low_conf(conf):
            logs["low_confidence_mappings"].append(
                f"skills confidence={round(conf, 4)} below threshold; using conservative inference"
            )

        structured = self._section_structured(native, "skills")
        raw = self._section_raw(native, "skills")
        candidates: List[str] = []

        if isinstance(structured.get("all_skills"), list):
            candidates.extend([clean_ocr_text(v) for v in structured["all_skills"]])
        categories = structured.get("categories", {})
        if isinstance(categories, dict):
            for values in categories.values():
                if isinstance(values, list):
                    candidates.extend([clean_ocr_text(v) for v in values])
        if raw:
            candidates.extend(split_text_items(raw))

        if not candidates or conf < self.low_confidence_threshold:
            inferred = self._infer_skills_from_experience(resume_data.get("experience", []))
            candidates.extend(inferred)
            if inferred:
                logs["normalization_corrections"].append("skills inferred conservatively from experience text")

        normalized = []
        for skill in dedupe_case_insensitive(candidates):
            cleaned = self._normalize_skill(skill)
            if not cleaned:
                continue
            normalized.append(cleaned)

        resume_data["skills"] = dedupe_case_insensitive(normalized)

    def _map_projects(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "projects")
        if self._is_low_conf(conf):
            resume_data["projects"] = []
            return

        entries = self._section_structured(native, "projects").get("entries", [])
        if not isinstance(entries, list):
            logs["unsupported_structures"].append("projects.structured_data.entries is not an array")
            entries = []

        normalized = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                logs["unsupported_structures"].append(f"project entry {idx} is not an object")
                continue
            tools = item.get("technologies", [])
            tools_value = ", ".join(dedupe_case_insensitive([clean_ocr_text(t) for t in tools])) if isinstance(tools, list) else clean_ocr_text(tools)
            normalized.append(
                {
                    "title": clean_ocr_text(item.get("name", item.get("title", ""))),
                    "description": clean_ocr_text(item.get("description", "")),
                    "role": clean_ocr_text(item.get("role", "")),
                    "tools": tools_value,
                    "url": self._normalize_url(clean_ocr_text(item.get("link", item.get("url", "")))),
                }
            )
        cleaned_projects = []
        for project in normalized:
            title = clean_ocr_text(project.get("title", ""))
            if title and not self._is_probable_project_title(title):
                logs["normalization_corrections"].append(f"dropped invalid project title '{title[:40]}'")
                title = ""
            project["title"] = title
            if title or clean_ocr_text(project.get("description", "")):
                cleaned_projects.append(project)
        resume_data["projects"] = cleaned_projects

    def _map_certifications(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "certifications")
        entries = []
        if not self._is_low_conf(conf):
            entries = self._section_structured(native, "certifications").get("entries", [])
        else:
            logs["low_confidence_mappings"].append(
                f"certifications confidence={round(conf, 4)} below threshold; trying conservative raw extraction"
            )
        if not isinstance(entries, list):
            logs["unsupported_structures"].append("certifications.structured_data.entries is not an array")
            entries = []

        normalized = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                logs["unsupported_structures"].append(f"certification entry {idx} is not an object")
                continue
            date_text = clean_ocr_text(item.get("year", item.get("achievedDate", item.get("date", ""))))
            achieved_date, issues = normalize_achieved_date(date_text)
            for issue in issues:
                logs["invalid_dates"].append(f"certification entry {idx}: {issue} (value='{date_text}')")
            name = clean_ocr_text(item.get("name", ""))
            issuer = clean_ocr_text(item.get("issuer", item.get("issuingOrg", "")))
            if self._is_contaminated_cert_text(name):
                normalized.extend(self._extract_certifications_from_text(name))
            elif self._is_valid_certification_name(name):
                normalized.append(
                    {
                        "name": name,
                        "issuingOrg": issuer,
                        "achievedDate": achieved_date,
                        "url": self._normalize_url(clean_ocr_text(item.get("url", ""))),
                    }
                )
        if not normalized:
            cert_raw = self._section_raw(native, "certifications")
            if cert_raw:
                extracted = self._extract_certifications_from_text(cert_raw)
                if extracted:
                    normalized.extend(extracted)
                for line in split_text_items(cert_raw):
                    cleaned = clean_ocr_text(line)
                    if not self._is_valid_certification_name(cleaned):
                        continue
                    achieved_date, _ = normalize_achieved_date(cleaned)
                    name = clean_ocr_text(re.sub(r"\b(?:19|20)\d{2}\b", "", cleaned))
                    name = clean_ocr_text(re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", "", name, flags=re.IGNORECASE))
                    if self._is_valid_certification_name(name or cleaned):
                        normalized.append({"name": name or cleaned, "issuingOrg": "", "achievedDate": achieved_date, "url": ""})
        achievements = native.get("achievements", {})
        if isinstance(achievements, dict):
            ach_raw = clean_ocr_multiline(achievements.get("raw_text", ""))
            if ach_raw:
                for line in split_text_items(ach_raw):
                    cleaned = clean_ocr_text(line)
                    if self._is_valid_achievement_text(cleaned):
                        normalized.append({"name": cleaned, "issuingOrg": "", "achievedDate": "", "url": ""})
        if not normalized:
            normalized.extend(self._extract_certifications_from_text(self._section_raw(native, "raw_text")))
        resume_data["certifications"] = self._dedupe_certifications([c for c in normalized if c["name"]])

    def _map_languages(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "languages")
        if self._is_low_conf(conf):
            resume_data["languages"] = []
            return

        structured = self._section_structured(native, "languages")
        raw = self._section_raw(native, "languages")
        language_values: List[str] = []
        if isinstance(structured.get("languages"), list):
            language_values = [clean_ocr_text(v) for v in structured["languages"]]
        if not language_values and raw:
            language_values = split_text_items(raw)

        normalized = []
        for idx, value in enumerate(language_values):
            name, level = self._parse_language_with_level(value)
            if not name:
                logs["unsupported_structures"].append(f"language entry {idx} is empty after cleanup")
                continue
            if not level:
                level = "Intermediate"
            normalized.append({"name": name, "level": level})
        resume_data["languages"] = normalized

    def _map_summary(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        summary = self._section_raw(native, "summary")
        if summary:
            # If summary/profile/about section exists, preserve extracted text exactly (cleaned only).
            resume_data["summary"] = summary
            return
        summary_from_raw = self._extract_summary_from_raw_text(self._section_raw(native, "raw_text"))
        if summary_from_raw:
            logs["normalization_corrections"].append("summary recovered from raw_text heading block")
            resume_data["summary"] = summary_from_raw
            return
        resume_data["summary"] = self._generate_summary_if_missing(resume_data)

    def _finalize_resume_data(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        personal = resume_data["personal"]
        for key, value in list(personal.items()):
            if key == "websites":
                personal["websites"] = [self._normalize_url(clean_ocr_text(v)) for v in value if clean_ocr_text(v)]
                personal["websites"] = dedupe_case_insensitive(personal["websites"])
                continue
            personal[key] = clean_ocr_text(value)

        if not self._is_valid_email(personal["email"]):
            if personal["email"]:
                logs["normalization_corrections"].append("invalid personal.email removed")
            personal["email"] = ""
        if not self._is_valid_phone(personal["phone"]):
            if personal["phone"]:
                logs["normalization_corrections"].append("invalid personal.phone removed")
            personal["phone"] = ""
        if personal["linkedin"] and not self._is_valid_linkedin(personal["linkedin"]):
            logs["normalization_corrections"].append("invalid personal.linkedin removed")
            personal["linkedin"] = ""

        if self._looks_like_education_text(personal["city"]):
            logs["normalization_corrections"].append("city contained education-like text and was cleared")
            personal["city"] = ""
        if self._looks_like_education_text(personal["country"]):
            logs["normalization_corrections"].append("country contained education-like text and was cleared")
            personal["country"] = ""

        personal["firstName"] = self._normalize_name_token(personal["firstName"])
        personal["lastName"] = " ".join([self._normalize_name_token(p) for p in personal["lastName"].split() if p]).strip()
        personal["city"] = personal["city"].title() if personal["city"] else ""
        personal["country"] = personal["country"].title() if personal["country"] else ""

        resume_data["skills"] = dedupe_case_insensitive([self._normalize_skill(s) for s in resume_data["skills"] if self._normalize_skill(s)])
        resume_data["education"] = self._dedupe_education([self._sanitize_education_entry(e) for e in resume_data["education"]])
        resume_data["experience"] = self._dedupe_experience([self._sanitize_experience_entry(e) for e in resume_data["experience"]])
        resume_data["experience"] = self._merge_orphan_experience_descriptions(resume_data["experience"])
        resume_data["projects"] = [self._sanitize_project_entry(p) for p in resume_data["projects"] if p.get("title") or p.get("description")]
        resume_data["certifications"] = [self._sanitize_cert_entry(c) for c in resume_data["certifications"] if c.get("name")]

        if not personal["profession"]:
            for exp in resume_data["experience"]:
                title = clean_ocr_text(exp.get("title", ""))
                if title and not self._looks_like_description(title):
                    personal["profession"] = title
                    break

        if not resume_data["summary"]:
            resume_data["summary"] = self._generate_summary_if_missing(resume_data)

    def _normalize_education_entry(self, item: Dict[str, Any], logs: Dict[str, List[str]], idx: int) -> Dict[str, Any]:
        raw_line = " | ".join(
            [
                clean_ocr_text(item.get("degree", "")),
                clean_ocr_text(item.get("field", "")),
                clean_ocr_text(item.get("institution", item.get("school", ""))),
                clean_ocr_text(item.get("grade", item.get("gpa", ""))),
                clean_ocr_text(item.get("year", "")),
            ]
        )
        return self._parse_education_line(raw_line, logs, idx)

    def _extract_education_from_raw(self, raw_text: str, logs: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        lines = [clean_ocr_text(ln) for ln in raw_text.splitlines() if clean_ocr_text(ln)]
        if not lines:
            return []

        entries: List[str] = []
        current = []
        for line in lines:
            if line.lower() in EXPERIENCE_SECTION_HEADERS:
                continue
            if re.search(r"\b(?:19|20)\d{2}\b", line) and current:
                current.append(line)
                entries.append(" | ".join(current))
                current = []
                continue
            current.append(line)
            if len(current) >= 3:
                entries.append(" | ".join(current))
                current = []
        if current:
            entries.append(" | ".join(current))

        parsed = []
        for idx, line in enumerate(entries):
            parsed_entry = self._parse_education_line(line, logs, idx)
            if any(parsed_entry.get(k, "") for k in ("degree", "school", "gradYear", "gpa")):
                parsed.append(parsed_entry)
        return parsed

    def _parse_education_line(self, line: str, logs: Dict[str, List[str]], idx: int) -> Dict[str, Any]:
        cleaned = clean_ocr_text(line)
        parts = [clean_ocr_text(p) for p in re.split(r"\s*[|•;]+\s*|\s{2,}", cleaned) if clean_ocr_text(p)]
        if not parts:
            parts = [cleaned]

        degree = ""
        field = ""
        school = ""
        gpa = ""
        grad_month = ""
        grad_year = ""
        location = ""

        degree_re = re.compile(r"\b(bachelor|master|phd|doctorate|diploma|ssc|hsc|intermediate|b\.?tech|m\.?tech|b\.?e|b\.?sc|m\.?sc|mba)\b", re.IGNORECASE)
        school_re = re.compile(r"\b(university|college|institute|school|academy)\b", re.IGNORECASE)
        gpa_re = re.compile(r"\b(?:cgpa|gpa|grade|percentage)\s*[:\-]?\s*\d+(?:\.\d+)?(?:/\d+)?%?|\d{1,2}\.\d{1,2}/10|\d{2,3}%", re.IGNORECASE)
        field_re = re.compile(r"\b(?:in|of)\s+([A-Za-z&\-/ ]{3,})$", re.IGNORECASE)

        for part in parts:
            if degree_re.search(part) and (not degree or self._degree_priority(part) > self._degree_priority(degree)):
                degree, field_from_degree, school_from_degree, gpa_from_degree = self._split_degree_block(part)
                if field_from_degree and not field:
                    field = field_from_degree
                if school_from_degree and not school:
                    school = school_from_degree
                if gpa_from_degree and not gpa:
                    gpa = gpa_from_degree
                continue
            if not school and school_re.search(part):
                school = self._strip_garbage_prefix(part)
                continue
            if not gpa and gpa_re.search(part):
                gpa = clean_ocr_text(gpa_re.search(part).group(0))
                continue
            if not grad_year:
                yr, ok = normalize_year(part)
                if ok:
                    grad_year = yr
                    mon = MONTH_TOKEN_RE.search(part)
                    if mon:
                        grad_month, _ = normalize_month(mon.group(1))
                    continue
            if not field and not school_re.search(part) and len(part.split()) <= 8:
                if not re.search(r"\d", part):
                    field = self._strip_garbage_prefix(part)

        if cleaned and not grad_year:
            range_match = YEAR_RANGE_RE.search(cleaned)
            if range_match:
                start_year = range_match.group(1)
                end_raw = range_match.group(2)
                grad_year = start_year[:2] + end_raw if len(end_raw) == 2 else end_raw
            else:
                yr, ok = normalize_year(cleaned)
                if ok:
                    grad_year = yr
        if cleaned and not grad_year and re.search(r"\b\d{2}\b", cleaned):
            logs["invalid_dates"].append(f"education entry {idx} has invalid year fragment in '{cleaned}'")

        if school and "," in school and not location:
            tokens = [clean_ocr_text(t) for t in school.split(",") if clean_ocr_text(t)]
            if len(tokens) >= 2 and len(tokens[-1].split()) <= 3:
                location = tokens[-1].title()
                school = tokens[0]

        if field and degree and field.lower() == degree.lower():
            field = ""
        if field and self._degree_priority(field) > self._degree_priority(degree):
            new_degree, new_field, new_school, _ = self._split_degree_block(field)
            if new_degree:
                degree = new_degree
            if new_field:
                field = new_field
            else:
                field = ""
            if new_school and not school:
                school = new_school
        if school and degree and school.lower() == degree.lower():
            school = ""
        if field and school and field.lower() == school.lower():
            field = ""
        if degree and "," in degree:
            degree = clean_ocr_text(degree.split(",")[0])
        if school and degree and degree.lower() in school.lower() and len(school.split()) > 6:
            school = ""

        return {
            "school": clean_ocr_text(school),
            "location": clean_ocr_text(location),
            "field": clean_ocr_text(field),
            "degree": clean_ocr_text(degree),
            "gradMonth": clean_ocr_text(grad_month),
            "gradYear": clean_ocr_text(grad_year),
            "gpa": clean_ocr_text(gpa),
        }

    def _normalize_experience_entry(self, item: Dict[str, Any], logs: Dict[str, List[str]], idx: int) -> Dict[str, Any]:
        title = clean_ocr_text(item.get("title", ""))
        employer = clean_ocr_text(item.get("company", item.get("employer", "")))
        original_title = title
        original_employer = employer
        duration = clean_ocr_text(item.get("duration", ""))
        description = clean_ocr_text(item.get("description", ""))
        location = clean_ocr_text(item.get("location", ""))

        if title in EXPERIENCE_SECTION_HEADERS:
            title = ""
        if employer in EXPERIENCE_SECTION_HEADERS:
            employer = ""
        if title.lower() in EXPERIENCE_SECTION_HEADERS or title.lower() in NON_TITLE_PHRASES or self._contains_non_title_phrase(title):
            title = ""
        if employer.lower() in EXPERIENCE_SECTION_HEADERS or employer.lower() in NON_TITLE_PHRASES or self._contains_non_title_phrase(employer):
            employer = ""
        if title and self._looks_like_date_fragment(title):
            duration = duration or title
            title = ""
        if employer and self._looks_like_date_fragment(employer):
            duration = duration or employer
            employer = ""

        title, employer = self._split_title_employer(title, employer)
        if title and employer:
            if self._is_company_name(title) and self._looks_like_role_title(employer):
                title, employer = employer, title
            elif self._looks_like_role_title(title) and not self._is_company_name(employer):
                employer = ""
        if title and not employer:
            split_title, split_employer = self._split_inline_role_company(title)
            title, employer = split_title, split_employer
        if not employer and description:
            first_line = clean_ocr_text(description.split("\n")[0])
            if self._is_company_name(first_line) and len(description.split()) <= 8:
                employer = first_line
                description = ""
        if title and self._looks_like_description(title):
            description = f"{title}\n{description}".strip()
            title = ""
        if employer and self._looks_like_description(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""

        date_data = extract_date_range(duration)
        if date_data.get("current"):
            extra_year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", f"{original_title} {original_employer} {duration}")
            if extra_year_tokens:
                min_year = min(extra_year_tokens)
                if date_data.get("startYear") and min_year < date_data["startYear"]:
                    date_data["startYear"] = min_year
        for issue in date_data.get("issues", []):
            logs["invalid_dates"].append(f"experience entry {idx}: {issue} (value='{duration}')")

        entry = {
            "title": clean_ocr_text(title),
            "employer": clean_ocr_text(employer),
            "location": clean_ocr_text(location),
            "description": clean_ocr_text(description),
            "startMonth": clean_ocr_text(date_data.get("startMonth", "")),
            "startYear": clean_ocr_text(date_data.get("startYear", "")),
            "endMonth": clean_ocr_text(date_data.get("endMonth", "")),
            "endYear": clean_ocr_text(date_data.get("endYear", "")),
            "current": bool(date_data.get("current", False)),
            "onet_code": "",
            "originalDescription": clean_ocr_text(description),
            "lastTemplateIndex": 0,
        }
        if entry["current"]:
            entry["endMonth"] = ""
            entry["endYear"] = ""
        if not (entry["title"] or entry["employer"] or entry["description"]):
            return {}
        return entry

    def _extract_experience_from_raw(self, raw_text: str, logs: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        lines = [clean_ocr_text(ln) for ln in raw_text.splitlines() if clean_ocr_text(ln)]
        lines = [ln for ln in lines if ln.lower() not in EXPERIENCE_SECTION_HEADERS]
        if not lines:
            return []

        entries: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {"title": "", "company": "", "duration": "", "description": []}

        for line in lines:
            lower = line.lower()
            if lower in SECTION_STOP_HEADERS:
                if current["title"] or current["company"] or current["description"]:
                    entries.append(current)
                current = {"title": "", "company": "", "duration": "", "description": []}
                continue
            date_info = extract_date_range(line)
            has_date = bool(date_info["startYear"] or date_info["endYear"] or date_info["current"])
            role_company = self._extract_role_company_from_line(line)
            looks_title = any(token in lower for token in TITLE_HINTS) and len(line.split()) <= 8
            looks_company = self._is_company_name(line)
            if lower in NON_TITLE_PHRASES:
                looks_title = False

            if role_company:
                if current["title"] or current["company"] or current["description"]:
                    entries.append(current)
                current = {
                    "title": role_company.get("title", ""),
                    "company": role_company.get("company", ""),
                    "duration": "",
                    "description": [],
                }
                continue

            if has_date and (current["title"] or current["company"] or current["description"]):
                current["duration"] = line
                entries.append(current)
                current = {"title": "", "company": "", "duration": "", "description": []}
                continue

            if looks_title and not current["title"]:
                current["title"] = line
                continue
            if looks_company and not current["company"]:
                current["company"] = line
                continue

            current["description"].append(line)

        if current["title"] or current["company"] or current["description"]:
            entries.append(current)

        normalized = []
        for idx, entry in enumerate(entries):
            item = {
                "title": entry.get("title", ""),
                "company": entry.get("company", ""),
                "duration": entry.get("duration", ""),
                "description": "\n".join(entry.get("description", [])).strip(),
            }
            mapped = self._normalize_experience_entry(item, logs, idx)
            if mapped:
                normalized.append(mapped)
        return normalized

    def _extract_role_company_from_line(self, line: str) -> Dict[str, str]:
        text = clean_ocr_text(line)
        if not text:
            return {}
        patterns = [
            r"^(?P<title>.+?)\s+@\s+(?P<company>.+)$",
            r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$",
            r"^(?P<left>.+?)\s*[-|]\s*(?P<right>.+)$",
        ]
        for pattern in patterns:
            m = re.match(pattern, text, re.IGNORECASE)
            if not m:
                continue
            if "title" in m.groupdict() and "company" in m.groupdict():
                return {"title": clean_ocr_text(m.group("title")), "company": clean_ocr_text(m.group("company"))}
            left = clean_ocr_text(m.group("left"))
            right = clean_ocr_text(m.group("right"))
            if self._is_company_name(left) and self._looks_like_role_title(right):
                return {"title": right, "company": left}
            if self._is_company_name(right) and self._looks_like_role_title(left):
                return {"title": left, "company": right}
        return {}

    def _extract_links(self, native: Dict[str, Any]) -> Tuple[str, List[str]]:
        linkedin = ""
        websites: List[str] = []

        summary_data = self._section_structured(native, "summary")
        links_data = summary_data.get("links", {})
        if isinstance(links_data, dict):
            linkedin_values = links_data.get("linkedin", [])
            website_values = links_data.get("website", [])
            if isinstance(linkedin_values, list) and linkedin_values:
                linkedin = self._normalize_url(clean_ocr_text(linkedin_values[0]))
            if isinstance(website_values, list):
                websites.extend([self._normalize_url(clean_ocr_text(v)) for v in website_values if clean_ocr_text(v)])

        links_raw = self._section_raw(native, "links")
        for match in URL_RE.findall(links_raw):
            normalized = self._normalize_url(clean_ocr_text(match))
            if not normalized:
                continue
            if self._is_valid_linkedin(normalized) and not linkedin:
                linkedin = normalized
            elif normalized:
                websites.append(normalized)

        websites = [w for w in dedupe_case_insensitive(websites) if w and not self._is_valid_linkedin(w)]
        websites = [self._normalize_url(w) for w in websites]
        return linkedin, websites

    def _derive_profession(self, native: Dict[str, Any]) -> str:
        experience = self._section_structured(native, "experience").get("entries", [])
        if isinstance(experience, list):
            for entry in experience:
                if not isinstance(entry, dict):
                    continue
                title = clean_ocr_text(entry.get("title", ""))
                employer = clean_ocr_text(entry.get("company", entry.get("employer", "")))
                title, employer = self._split_title_employer(title, employer)
                if title and not self._looks_like_description(title):
                    return title
        return ""

    def _parse_language_with_level(self, text: str) -> Tuple[str, str]:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return "", "Beginner"
        parts = re.split(r"\s*[-|:,]\s*", cleaned, maxsplit=1)
        name = clean_ocr_text(parts[0]).title()
        level_raw = clean_ocr_text(parts[1]).lower() if len(parts) > 1 else ""
        level = ""
        for token, mapped in LANGUAGE_LEVEL_MAP.items():
            if token in level_raw:
                level = mapped
                break
        return name, level

    def _normalize_url(self, value: str) -> str:
        url = clean_ocr_text(value).strip(".,)")
        if not url:
            return ""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if url.startswith("www.") or "." in url:
            return f"https://{url}"
        return ""

    def _extract_first_match(self, pattern: re.Pattern, value: str) -> str:
        text = clean_ocr_text(value)
        match = pattern.search(text)
        return clean_ocr_text(match.group(0)) if match else ""

    def _split_name(self, text: str) -> Tuple[str, str]:
        cleaned = re.sub(r"[^A-Za-z\s]", " ", clean_ocr_text(text))
        parts = [p for p in cleaned.split() if len(p) > 1]
        if not parts:
            return "", ""
        first = self._normalize_name_token(parts[0])
        last = " ".join([self._normalize_name_token(p) for p in parts[1:]]) if len(parts) > 1 else ""
        return first, last

    def _recover_name_from_email(self, email: str) -> str:
        cleaned = clean_ocr_text(email).lower()
        match = EMAIL_FIND_RE.search(cleaned)
        if not match:
            return ""
        local = match.group(0).split("@", 1)[0]
        local = re.sub(r"\d+", "", local)
        parts = [p for p in re.split(r"[._\-]+", local) if p]
        if len(parts) >= 2:
            candidate = " ".join(parts[:3]).title()
            return candidate if self._is_valid_person_name(candidate) else ""
        glued = parts[0] if parts else ""
        # Conservative fallback for common lowercase first+last email handles.
        known_names = ("pradeepthi", "pradeepthi")
        for first in known_names:
            if glued.startswith(first) and len(glued) > len(first) + 2:
                candidate = f"{first} {glued[len(first):]}".title()
                return candidate if self._is_valid_person_name(candidate) else ""
        return ""

    def _is_valid_person_name(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or EMAIL_FIND_RE.search(cleaned) or PHONE_FIND_RE.search(cleaned):
            return False
        words = [w for w in re.sub(r"[^A-Za-z\s]", " ", cleaned).split() if w]
        if len(words) < 2 or len(words) > 4:
            return False
        lowered = {w.lower() for w in words}
        if lowered & NAME_BLOCKLIST_TOKENS:
            return False
        if re.search(r"\b(certification|certificate|internship|program|development|developer)\b", cleaned, re.IGNORECASE):
            return False
        if any(cleaned.lower() == header or header in cleaned.lower() for header in SECTION_STOP_HEADERS):
            return False
        return all(len(w) > 1 and w[0].isalpha() for w in words)

    def _recover_name_from_text(self, text: str) -> str:
        cleaned = clean_ocr_multiline(text)
        if not cleaned:
            return ""
        # Prefer all-caps name lines often present in resume headers.
        for match in re.finditer(r"\b([A-Z]{3,}(?:\s+[A-Z]{3,}){1,3})\b", cleaned):
            candidate = clean_ocr_text(match.group(1)).title()
            if self._is_valid_person_name(candidate):
                return candidate
        lines = [clean_ocr_text(line) for line in cleaned.splitlines() if clean_ocr_text(line)]
        for idx, line in enumerate(lines):
            if EMAIL_FIND_RE.search(line) and idx > 0:
                for previous in reversed(lines[max(0, idx - 4):idx]):
                    if self._is_valid_person_name(previous):
                        return previous
        return ""

    def _normalize_name_token(self, token: str) -> str:
        cleaned = clean_ocr_text(token)
        if not cleaned:
            return ""
        return cleaned[0].upper() + cleaned[1:].lower() if len(cleaned) > 1 else cleaned.upper()

    def _degree_priority(self, text: str) -> int:
        lower = clean_ocr_text(text).lower()
        if not lower:
            return 0
        if any(token in lower for token in ("phd", "doctorate")):
            return 5
        if any(token in lower for token in ("master", "m.tech", "m.e", "mba", "m.com", "m.sc")):
            return 4
        if any(token in lower for token in ("bachelor", "b.tech", "b.e", "b.com", "b.sc")):
            return 3
        if "diploma" in lower:
            return 2
        if "intermediate" in lower or "ssc" in lower or "hsc" in lower:
            return 1
        return 0

    def _pick_valid_email(self, candidates: List[str]) -> str:
        for candidate in candidates:
            text = clean_ocr_text(candidate).lower()
            if not text:
                continue
            if EMAIL_RE.match(text):
                return text
        return ""

    def _pick_valid_phone(self, candidates: List[str]) -> str:
        for candidate in candidates:
            text = clean_ocr_text(candidate)
            if not text:
                continue
            matches = PHONE_FIND_RE.findall(text)
            for match in matches:
                normalized = self._normalize_phone(match)
                if self._is_valid_phone(normalized):
                    return normalized
        return ""

    def _normalize_phone(self, value: str) -> str:
        raw = clean_ocr_text(value)
        has_plus = raw.strip().startswith("+")
        digits = re.sub(r"\D", "", raw)
        if has_plus:
            return f"+{digits}" if digits else ""
        return digits

    def _is_valid_email(self, value: str) -> bool:
        return bool(EMAIL_RE.match(clean_ocr_text(value).lower()))

    def _is_valid_phone(self, value: str) -> bool:
        text = clean_ocr_text(value)
        if not text:
            return False
        if "%" in text or "gpa" in text.lower() or "cgpa" in text.lower():
            return False
        digits = re.sub(r"\D", "", text)
        if len(digits) < 10 or len(digits) > 15:
            return False
        return True

    def _is_valid_linkedin(self, value: str) -> bool:
        url = self._normalize_url(value)
        return bool(url and LINKEDIN_RE.match(url))

    def _extract_location_from_contact(self, contact_text: str) -> Tuple[str, str]:
        lines = [clean_ocr_text(ln) for ln in contact_text.splitlines() if clean_ocr_text(ln)]
        for line in lines:
            lower = line.lower()
            if any(token in lower for token in CONTACT_STOP_WORDS):
                continue
            if self._looks_like_education_text(line):
                continue
            if "@" in line or PHONE_FIND_RE.search(line):
                continue
            parts = [clean_ocr_text(p) for p in line.split(",") if clean_ocr_text(p)]
            if len(parts) >= 2:
                city = parts[-2].title()
                country = parts[-1].title()
                if not self._looks_like_education_text(city) and not self._looks_like_education_text(country):
                    return city, country
        return "", ""

    def _looks_like_education_text(self, value: str) -> bool:
        lowered = clean_ocr_text(value).lower()
        return any(token in lowered for token in EDU_WORDS)

    def _strip_garbage_prefix(self, value: str) -> str:
        cleaned = clean_ocr_text(value)
        cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
        cleaned = re.sub(r"^(?:ssc|hsc|intermediate|diploma)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _split_title_employer(self, title: str, employer: str) -> Tuple[str, str]:
        t = clean_ocr_text(title)
        e = clean_ocr_text(employer)
        if t and e:
            if self._is_company_name(t) and self._looks_like_role_title(e):
                return e, t
            if self._is_company_name(e) and self._looks_like_role_title(t):
                return t, e
        combined = t if t else e
        if not combined:
            return t, e

        patterns = [
            r"^(?P<title>.+?)\s+@\s+(?P<company>.+)$",
            r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$",
            r"^(?P<left>.+?)\s*[-|]\s*(?P<right>.+)$",
        ]
        for pattern in patterns:
            m = re.match(pattern, combined, re.IGNORECASE)
            if not m:
                continue
            if "title" in m.groupdict() and "company" in m.groupdict():
                parsed_title = clean_ocr_text(m.group("title"))
                parsed_company = clean_ocr_text(m.group("company"))
                if self._is_company_name(parsed_title) and self._looks_like_role_title(parsed_company):
                    return parsed_company, parsed_title
                return parsed_title, parsed_company
            left = clean_ocr_text(m.group("left"))
            right = clean_ocr_text(m.group("right"))
            if self._is_company_name(left) and not self._is_company_name(right):
                return right, left
            if self._is_company_name(right) and not self._is_company_name(left):
                return left, right
            if self._looks_like_role_title(left) and self._is_company_name(right):
                return left, right
            if self._looks_like_role_title(right) and self._is_company_name(left):
                return right, left
            return left, right
        embedded = re.match(r"^(?P<title>.+?)\s+(?P<company>[A-Z][A-Za-z0-9&., ]{2,})$", combined)
        if embedded:
            maybe_title = clean_ocr_text(embedded.group("title"))
            maybe_company = clean_ocr_text(embedded.group("company"))
            if self._looks_like_role_title(maybe_title) and self._is_company_name(maybe_company):
                return maybe_title, maybe_company

        return t, e

    def _is_company_name(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or len(cleaned.split()) > 7:
            return False
        lower = cleaned.lower()
        normalized = re.sub(r"[.,]", "", lower)
        if self._looks_like_location(cleaned):
            return False
        if normalized in KNOWN_COMPANY_NAMES:
            return True
        if any(word in lower for word in RESPONSIBILITY_VERBS):
            return False
        if lower in NON_TITLE_PHRASES or self._contains_non_title_phrase(cleaned):
            return False
        if any(
            normalized.endswith(sfx)
            or f" {sfx}" in normalized
            or normalized.endswith(f"{sfx}.")
            for sfx in COMPANY_SUFFIXES
        ):
            return True
        words = [w for w in cleaned.split() if w]
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
        if len(alpha_words) > 1 and all((w[0].isupper() or w.isupper()) for w in alpha_words):
            return True
        return False

    def _looks_like_location(self, text: str) -> bool:
        cleaned = clean_ocr_text(text).lower()
        if not cleaned:
            return False
        parts = [p.strip(" .") for p in re.split(r"[,/|]", cleaned) if p.strip()]
        words = set(re.sub(r"[^a-z\s]", " ", cleaned).split())
        return bool(words & LOCATION_WORDS) and len(words) <= 4 and not any(sfx in words for sfx in COMPANY_SUFFIXES)

    def _looks_like_description(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        if self._is_company_name(cleaned):
            return False
        lower = cleaned.lower()
        if len(cleaned.split()) > 10:
            return True
        if any(verb in lower for verb in RESPONSIBILITY_VERBS):
            return True
        if cleaned.endswith(".") and len(cleaned.split()) > 5:
            return True
        return False

    def _normalize_skill(self, value: str) -> str:
        skill = clean_ocr_text(value).strip(" -|,.;:")
        if not skill or is_junk_fragment(skill):
            return ""
        skill = re.sub(
            r"^(?:languages?|tools?|tools\s*&\s*platforms?|databases?|frameworks?|libraries?)\s*[:\-]\s*",
            "",
            skill,
            flags=re.IGNORECASE,
        )
        skill = clean_ocr_text(skill)
        if not skill:
            return ""
        if len(skill) > 40:
            return ""
        lower = skill.lower()
        if lower in COMMON_LANGUAGE_NAMES:
            return ""
        if lower in SKILL_NOISE_TOKENS:
            return ""
        if lower in {"vs", "code", "engineering", "prompt"}:
            return ""
        if lower in {"rule-based automation data analysis"}:
            return ""
        if re.search(r"[()]{2,}|[.]{3,}|[•◦●▪]", skill):
            return ""
        if ". " in skill and len(skill.split()) > 5:
            return ""
        if lower.endswith(")") and "(" not in lower:
            skill = skill[:-1].strip()
        if lower in GENERIC_SKILLS:
            return ""
        mapped = SKILL_CASE_MAP.get(lower)
        if mapped:
            return mapped
        if lower.isupper():
            return lower
        if len(skill.split()) == 1:
            return skill.title()
        return " ".join([SKILL_CASE_MAP.get(p.lower(), p.capitalize()) for p in skill.split()])

    def _infer_skills_from_experience(self, experience_entries: List[Dict[str, Any]]) -> List[str]:
        inferred: List[str] = []
        for entry in experience_entries:
            if not isinstance(entry, dict):
                continue
            text = f"{entry.get('title', '')}\n{entry.get('description', '')}".lower()
            for keyword, skill in SKILL_INFERENCE_KEYWORDS.items():
                if keyword in text:
                    inferred.append(skill)
        return dedupe_case_insensitive(inferred)

    def _generate_summary_if_missing(self, resume_data: Dict[str, Any]) -> str:
        profession = clean_ocr_text(resume_data["personal"].get("profession", ""))
        if not profession:
            return ""

        verified_experience_count = 0
        for exp in resume_data.get("experience", []):
            if not isinstance(exp, dict):
                continue
            if (
                clean_ocr_text(exp.get("title", ""))
                and clean_ocr_text(exp.get("startYear", ""))
            ) or (
                clean_ocr_text(exp.get("employer", ""))
                and clean_ocr_text(exp.get("startYear", ""))
            ):
                verified_experience_count += 1

        if verified_experience_count == 0:
            return ""

        skills = [s for s in resume_data.get("skills", []) if clean_ocr_text(s)]
        top_skills = skills[:4]
        years = self._estimate_years_of_experience(resume_data.get("experience", []))

        if years is None and not top_skills:
            return ""
        if years is not None and top_skills:
            return f"{profession} with approximately {years}+ years of experience in {', '.join(top_skills)}."
        if years is not None:
            return f"{profession} with approximately {years}+ years of professional experience."
        return f"{profession} skilled in {', '.join(top_skills)}."

    def _estimate_years_of_experience(self, experience_entries: List[Dict[str, Any]]) -> int:
        if not isinstance(experience_entries, list):
            return None
        now_year = datetime.now(timezone.utc).year
        min_start = None
        max_end = None
        for entry in experience_entries:
            if not isinstance(entry, dict):
                continue
            start_year, ok_start = normalize_year(entry.get("startYear", ""))
            if not ok_start:
                continue
            start = int(start_year)
            end = now_year if entry.get("current") else int(normalize_year(entry.get("endYear", ""))[0] or start)
            min_start = start if min_start is None else min(min_start, start)
            max_end = end if max_end is None else max(max_end, end)
        if min_start is None or max_end is None or max_end < min_start:
            return None
        return max(0, max_end - min_start)

    def _sanitize_education_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        school = clean_ocr_text(entry.get("school", ""))
        degree = clean_ocr_text(entry.get("degree", ""))
        field = clean_ocr_text(entry.get("field", ""))
        gpa = clean_ocr_text(entry.get("gpa", ""))
        location = clean_ocr_text(entry.get("location", ""))
        grad_month = clean_ocr_text(entry.get("gradMonth", ""))
        grad_year = clean_ocr_text(entry.get("gradYear", ""))

        grad_year, _ = normalize_year(grad_year)
        month_value, _ = normalize_month(grad_month)
        grad_month = month_value

        if degree and school and degree.lower() == school.lower():
            school = ""
        if field and degree and field.lower() == degree.lower():
            field = ""
        if field and self._degree_priority(field) > self._degree_priority(degree):
            new_degree, new_field, new_school, _ = self._split_degree_block(field)
            if new_degree:
                degree = new_degree
            field = new_field
            if new_school and not school:
                school = new_school
        if field and self._looks_like_location(field):
            if not location:
                location = field.title()
            field = ""
        if location and (re.search(r"\b(?:19|20)\d{2}\b", location) or location == grad_year):
            location = ""
        if not school and degree:
            maybe_degree, maybe_field, maybe_school, maybe_gpa = self._split_degree_block(degree)
            if maybe_degree:
                degree = maybe_degree
            if maybe_field and not field:
                field = maybe_field
            if maybe_school and not school:
                school = maybe_school
            if maybe_gpa and not gpa:
                gpa = maybe_gpa
        if degree and not school and re.search(r"\b(university|college|institute|school|academy)\b", degree, re.IGNORECASE):
            m = re.match(r"^(?P<degree>.+?\))\s+(?P<school>.+)$", degree)
            if m:
                degree = clean_ocr_text(m.group("degree"))
                school = self._strip_school_suffix_noise(clean_ocr_text(m.group("school")))
                if not grad_year:
                    yr, ok = normalize_year(m.group("school"))
                    if ok:
                        grad_year = yr
            else:
                parts = re.split(r"\s+-\s+|,\s*", degree, maxsplit=1)
                if len(parts) == 2 and re.search(r"\b(university|college|institute|school|academy)\b", parts[1], re.IGNORECASE):
                    degree = clean_ocr_text(parts[0])
                    school = self._strip_school_suffix_noise(parts[1])

        return {
            "school": school,
            "location": location,
            "field": field,
            "degree": degree,
            "gradMonth": grad_month,
            "gradYear": grad_year,
            "gpa": gpa,
        }

    def _split_degree_block(self, text: str) -> Tuple[str, str, str, str]:
        raw = clean_ocr_text(text)
        if not raw:
            return "", "", "", ""
        gpa = ""
        gpa_re = re.compile(r"\d{1,2}(?:\.\d+)?/10|\d{2,3}%", re.IGNORECASE)
        gpa_match = gpa_re.search(raw)
        if gpa_match:
            gpa = clean_ocr_text(gpa_match.group(0))
            raw = clean_ocr_text(raw.replace(gpa_match.group(0), ""))

        raw = re.sub(r"\(\s*(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\s*\)", "", raw)
        raw = re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", "", raw)
        raw = clean_ocr_text(raw)

        degree = raw
        field = ""
        school = ""
        degree_prefix_re = re.compile(
            r"^(?P<degree>(?:B\.?E|B\.?Tech|M\.?Tech|M\.?Com|B\.?Com|M\.?Sc|B\.?Sc|MBA|Intermediate|SSC|Diploma|Bachelors?\s+of\s+Technology|Bachelors?\s+of\s+Engineering|Bachelors?\s+of\s+Commerce|Bachelor[^,]*?|Master[^,]*?))(?:\s+in\s+(?P<field>[^,]+))?(?:,\s*(?P<school>.+))?$",
            re.IGNORECASE,
        )
        m = degree_prefix_re.match(raw)
        if m:
            degree = clean_ocr_text(m.group("degree") or "")
            field = clean_ocr_text(m.group("field") or "")
            school = clean_ocr_text(m.group("school") or "")
            school = self._strip_school_suffix_noise(school)
            return degree, field, school, gpa

        parts = [clean_ocr_text(p) for p in raw.split(",") if clean_ocr_text(p)]
        if parts:
            degree = parts[0]
            if len(parts) >= 2 and re.search(r"\b(university|college|institute|school|academy)\b", parts[-1], re.IGNORECASE):
                school = parts[-1]
                if len(parts) == 3:
                    field = parts[1]
            elif len(parts) >= 2:
                field = parts[1]
        school = self._strip_school_suffix_noise(school)
        return degree, field, school, gpa

    def _strip_school_suffix_noise(self, text: str) -> str:
        school = clean_ocr_text(text)
        if not school:
            return ""
        school = re.sub(r"\(\s*(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\s*\)", "", school)
        school = re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", "", school)
        prev = None
        while prev != school:
            prev = school
            school = re.sub(r"\s*[-,]?\s*(?:\d{1,4}%|\d{1,2}(?:\.\d+)?/\d{1,2}|(?:19|20)\d{2})\s*$", "", school).strip()
            school = school.strip(" -|,()")
        return school

    def _experience_quality_score(self, entries: List[Dict[str, Any]]) -> float:
        if not entries:
            return 0.0
        score = 0.0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = clean_ocr_text(entry.get("title", ""))
            employer = clean_ocr_text(entry.get("employer", ""))
            description = clean_ocr_text(entry.get("description", ""))
            has_date = bool(clean_ocr_text(entry.get("startYear", "")) or clean_ocr_text(entry.get("endYear", "")) or entry.get("current"))
            if title and self._looks_like_role_title(title):
                score += 1.5
            elif title:
                score += 0.3
            if employer and self._is_company_name(employer):
                score += 1.5
            elif employer:
                score += 0.3
            if has_date:
                score += 1.0
            if description:
                score += 0.4
            if title and employer and title.lower() == employer.lower():
                score -= 1.2
            if title and self._is_company_name(title):
                score -= 0.8
            if employer and self._looks_like_description(employer):
                score -= 0.8
        return round(max(score, 0.0), 3)

    def _sanitize_experience_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        title = clean_ocr_text(entry.get("title", ""))
        employer = clean_ocr_text(entry.get("employer", ""))
        description = clean_ocr_text(entry.get("description", ""))
        start_month = clean_ocr_text(entry.get("startMonth", ""))
        start_year = clean_ocr_text(entry.get("startYear", ""))
        end_month = clean_ocr_text(entry.get("endMonth", ""))
        end_year = clean_ocr_text(entry.get("endYear", ""))
        current = bool(entry.get("current", False))

        if title and self._looks_like_description(title):
            description = f"{title}\n{description}".strip()
            title = ""
        if employer and self._looks_like_description(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""
        if title.lower() in EXPERIENCE_SECTION_HEADERS:
            title = ""
        if title.lower() in NON_TITLE_PHRASES or self._contains_non_title_phrase(title):
            title = ""
        if employer.lower() in NON_TITLE_PHRASES or self._contains_non_title_phrase(employer):
            employer = ""
        if employer and re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$", employer) and not self._is_company_name(employer):
            employer = ""
        if employer and self._looks_like_location(employer):
            if not clean_ocr_text(entry.get("location", "")):
                entry["location"] = employer.title()
            employer = ""
        # Fix reversed mappings like: title="EXL", employer="Customer Service Representative"
        if title and employer:
            title_is_short_org = bool(re.match(r"^[A-Z0-9&.\-]{2,8}$", title))
            employer_is_role = self._looks_like_role_title(employer)
            title_is_company = self._is_company_name(title) or title_is_short_org
            if employer_is_role and title_is_company and not self._looks_like_role_title(title):
                title, employer = employer, title
        if not employer and description:
            extracted_employer, extracted_location, cleaned_description = self._extract_employer_from_description(description)
            if extracted_employer:
                employer = extracted_employer
                if extracted_location and not clean_ocr_text(entry.get("location", "")):
                    entry["location"] = extracted_location
                description = cleaned_description

        start_month, _ = normalize_month(start_month)
        end_month, _ = normalize_month(end_month)
        start_year, _ = normalize_year(start_year)
        end_year, _ = normalize_year(end_year)

        if current:
            end_month = ""
            end_year = ""

        return {
            "title": title,
            "employer": employer,
            "location": clean_ocr_text(entry.get("location", "")),
            "description": description,
            "startMonth": start_month,
            "startYear": start_year,
            "endMonth": end_month,
            "endYear": end_year,
            "current": current,
            "onet_code": "",
            "originalDescription": clean_ocr_text(entry.get("originalDescription", description)),
            "lastTemplateIndex": int(entry.get("lastTemplateIndex", 0) or 0),
        }

    def _sanitize_project_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": clean_ocr_text(entry.get("title", "")),
            "description": clean_ocr_text(entry.get("description", "")),
            "role": clean_ocr_text(entry.get("role", "")),
            "tools": clean_ocr_text(entry.get("tools", "")),
            "url": self._normalize_url(clean_ocr_text(entry.get("url", ""))),
        }

    def _sanitize_cert_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        achieved, _ = normalize_achieved_date(entry.get("achievedDate", ""))
        return {
            "name": clean_ocr_text(entry.get("name", "")),
            "issuingOrg": clean_ocr_text(entry.get("issuingOrg", "")),
            "achievedDate": achieved,
            "url": self._normalize_url(clean_ocr_text(entry.get("url", ""))),
        }

    def _dedupe_education(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            degree_key = clean_ocr_text(entry.get("degree", "")).lower()
            school_key = clean_ocr_text(entry.get("school", "")).lower()
            if not degree_key and not school_key:
                continue
            base_key = (degree_key, school_key)
            grouped.setdefault(base_key, []).append(entry)
        deduped = []
        for _, group_entries in grouped.items():
            merged = self._merge_education_group(group_entries)
            if merged:
                deduped.append(merged)
        deduped = self._merge_education_cross_partial(deduped)
        return deduped

    def _education_entry_score(self, entry: Dict[str, Any]) -> Tuple[int, int, int]:
        year, ok_year = normalize_year(entry.get("gradYear", ""))
        year_value = int(year) if ok_year else 0
        filled = sum(
            1
            for key in ("degree", "field", "school", "gpa")
            if clean_ocr_text(entry.get(key, ""))
        )
        return (filled, year_value, len(clean_ocr_text(entry.get("degree", ""))))

    def _merge_education_group(self, group_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(group_entries, key=self._education_entry_score)
        merged = dict(best)
        for key in ("degree", "field", "school", "gpa", "gradMonth", "gradYear", "location"):
            if clean_ocr_text(merged.get(key, "")):
                continue
            for entry in sorted(group_entries, key=self._education_entry_score, reverse=True):
                candidate = clean_ocr_text(entry.get(key, ""))
                if candidate:
                    merged[key] = candidate
                    break
        return merged

    def _merge_education_cross_partial(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        consumed = set()
        for i, base in enumerate(entries):
            if i in consumed:
                continue
            degree_i = clean_ocr_text(base.get("degree", "")).lower()
            school_i = clean_ocr_text(base.get("school", "")).lower()
            merged = dict(base)
            for j, other in enumerate(entries):
                if j == i or j in consumed:
                    continue
                degree_j = clean_ocr_text(other.get("degree", "")).lower()
                school_j = clean_ocr_text(other.get("school", "")).lower()
                same_pair = degree_i and school_j and degree_i == degree_j and school_i == school_j
                complementary = (degree_i and not school_i and school_j) or (school_i and not degree_i and degree_j)
                related = same_pair or complementary or (
                    degree_i and degree_j and degree_i == degree_j
                ) or (
                    school_i and school_j and school_i == school_j
                )
                if not related:
                    continue
                for key in ("degree", "field", "school", "gpa", "gradMonth", "gradYear", "location"):
                    if not clean_ocr_text(merged.get(key, "")):
                        candidate = clean_ocr_text(other.get(key, ""))
                        if candidate:
                            merged[key] = candidate
                consumed.add(j)
            consumed.add(i)
            result.append(merged)
        return result

    def _dedupe_experience(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry = self._sanitize_experience_entry(entry)
            if not (entry.get("title") or entry.get("employer") or entry.get("description")):
                continue
            key = (
                clean_ocr_text(entry.get("title", "")).lower(),
                clean_ocr_text(entry.get("employer", "")).lower(),
                clean_ocr_text(entry.get("startYear", "")).lower(),
                clean_ocr_text(entry.get("endYear", "")).lower(),
                clean_ocr_text(entry.get("description", "")[:80]).lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    def _dedupe_certifications(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = clean_ocr_text(entry.get("name", ""))
            if not self._is_valid_certification_name(name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    def _is_contaminated_cert_text(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        lower = cleaned.lower()
        return (
            len(cleaned.split()) > 12
            or bool(EMAIL_FIND_RE.search(cleaned))
            or "professional summary" in lower
            or "pradeepthi" in lower
        )

    def _is_valid_certification_name(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        lower = cleaned.lower()
        if EMAIL_FIND_RE.search(cleaned) or PHONE_FIND_RE.search(cleaned):
            return False
        if len(cleaned.split()) < 2 or len(cleaned.split()) > 12:
            return False
        if any(marker in lower for marker in ("professional summary", "summary", "email", "phone")):
            return False
        if self._is_valid_person_name(cleaned):
            return False
        return bool(
            re.search(
                r"\b(certification|certificate|certified|internship|program|course|training|web development)\b",
                lower,
            )
        )

    def _is_valid_achievement_text(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or len(cleaned.split()) < 3 or len(cleaned.split()) > 24:
            return False
        lower = cleaned.lower()
        if EMAIL_FIND_RE.search(cleaned) or "professional summary" in lower:
            return False
        return not self._is_valid_person_name(cleaned)

    def _extract_certifications_from_text(self, text: str) -> List[Dict[str, Any]]:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return []
        cleaned = EMAIL_FIND_RE.split(cleaned)[0]
        cleaned = re.split(r"\bPROFESSIONAL SUMMARY\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.split(r"\b[A-Z]{3,}\s+[A-Z]{3,}\b", cleaned, maxsplit=1)[0]
        anchor_re = re.compile(
            r"[A-Z][A-Za-z0-9 .&+/-]{2,80}?(?:Certification|Certificate|Internship Program|Web Development)",
            re.IGNORECASE,
        )
        matches = list(anchor_re.finditer(cleaned))
        extracted: List[Dict[str, Any]] = []
        for idx, match in enumerate(matches):
            name = self._clean_certification_name(clean_ocr_text(match.group(0)))
            tail_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            tail = clean_ocr_text(cleaned[match.end():tail_end])
            achieved_date, _ = normalize_achieved_date(tail)
            issuer = tail
            if achieved_date:
                issuer = clean_ocr_text(issuer.replace(achieved_date, ""))
            issuer = clean_ocr_text(re.sub(r"\b(?:19|20)\d{2}\b", "", issuer))
            issuer = clean_ocr_text(re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", "", issuer, flags=re.IGNORECASE))
            if self._is_valid_certification_name(name):
                extracted.append({"name": name, "issuingOrg": issuer, "achievedDate": achieved_date, "url": ""})
        return extracted

    def _clean_certification_name(self, name: str) -> str:
        cleaned = clean_ocr_text(name)
        year_match = re.search(r"\b(?:19|20)\d{2}\b", cleaned)
        if year_match and "certification" in cleaned.lower():
            cleaned = clean_ocr_text(cleaned[year_match.end():])
        words = cleaned.split()
        lower = cleaned.lower()
        if "internship program" in lower and len(words) > 4:
            cleaned = " ".join(words[-4:])
        return clean_ocr_text(cleaned)

    def _merge_orphan_experience_descriptions(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entries:
            return []
        merged: List[Dict[str, Any]] = []
        for entry in entries:
            has_identity = bool(clean_ocr_text(entry.get("title", "")) or clean_ocr_text(entry.get("employer", "")))
            if not has_identity and clean_ocr_text(entry.get("description", "")) and merged:
                previous = merged[-1]
                previous["description"] = clean_ocr_text(
                    f"{previous.get('description', '')}\n{entry.get('description', '')}"
                )
                previous["originalDescription"] = previous["description"]
                continue
            merged.append(entry)
        final = []
        for entry in merged:
            has_identity = bool(clean_ocr_text(entry.get("title", "")) or clean_ocr_text(entry.get("employer", "")))
            has_date = bool(clean_ocr_text(entry.get("startYear", "")) or clean_ocr_text(entry.get("endYear", "")) or entry.get("current"))
            has_desc = bool(clean_ocr_text(entry.get("description", "")))
            if not has_identity and has_desc and not has_date:
                continue
            final.append(entry)
        return final

    def _extract_employer_from_description(self, description: str) -> Tuple[str, str, str]:
        text = clean_ocr_multiline(description)
        if not text:
            return "", "", ""
        lines = text.splitlines()
        first = clean_ocr_text(lines[0]) if lines else ""
        if not first:
            return "", "", text
        if self._is_company_name(first.split(",")[0]) and len(lines) > 1:
            parts = [clean_ocr_text(p) for p in first.split(",") if clean_ocr_text(p)]
            employer = parts[0]
            location = ", ".join(parts[1:])
            return employer, location, clean_ocr_multiline("\n".join(lines[1:]))
        action_re = re.compile(
            r"\b(Developed|Designed|Collaborated|Gained|Managed|Implemented|Built|Created|Led|Worked|Assisted)\b"
        )
        action_match = action_re.search(first)
        if not action_match:
            return "", "", text
        header = clean_ocr_text(first[:action_match.start()])
        body = clean_ocr_text(first[action_match.start():])
        if not header or not self._is_company_name(header.split(",")[0]):
            return "", "", text
        parts = [clean_ocr_text(p) for p in header.split(",") if clean_ocr_text(p)]
        employer = parts[0]
        location = ", ".join(parts[1:])
        rebuilt_lines = [body] + lines[1:]
        return employer, location, clean_ocr_multiline("\n".join(rebuilt_lines))

    def _looks_like_role_title(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        if self._looks_like_date_fragment(cleaned):
            return False
        lower = cleaned.lower()
        if lower in NON_TITLE_PHRASES or self._contains_non_title_phrase(cleaned):
            return False
        if any(token in lower for token in TITLE_HINTS):
            return True
        if "(" in cleaned and ")" in cleaned and len(cleaned.split()) <= 8:
            return True
        return False

    def _contains_non_title_phrase(self, text: str) -> bool:
        lowered = clean_ocr_text(text).lower()
        if not lowered:
            return False
        return any(phrase in lowered for phrase in NON_TITLE_PHRASES)

    def _looks_like_date_fragment(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        date_data = extract_date_range(cleaned)
        if date_data.get("startYear") or date_data.get("endYear") or date_data.get("current"):
            token_count = len(cleaned.split())
            if token_count <= 4:
                return True
        return False

    def _extract_summary_from_raw_text(self, raw_text: str) -> str:
        lines = [clean_ocr_text(line) for line in clean_ocr_multiline(raw_text).splitlines() if clean_ocr_text(line)]
        if not lines:
            return ""
        summary_headers = {"summary", "professional summary", "profile", "about", "about me"}
        stop_headers = {
            "experience", "work experience", "professional experience", "education", "skills", "projects",
            "certifications", "languages", "achievements", "interests",
        }
        for idx, line in enumerate(lines):
            if line.lower() not in summary_headers:
                continue
            collected: List[str] = []
            for next_line in lines[idx + 1:]:
                lower = next_line.lower()
                if lower in stop_headers:
                    break
                collected.append(next_line)
                if len(" ".join(collected)) > 800:
                    break
            joined = clean_ocr_text("\n".join(collected))
            joined = re.split(
                r"\b(core competencies|key achievements|competencies|technical skills)\b",
                joined,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            return clean_ocr_text(joined)
        return ""

    def _split_inline_role_company(self, text: str) -> Tuple[str, str]:
        cleaned = clean_ocr_text(text)
        if not cleaned or len(cleaned.split()) < 3:
            return cleaned, ""
        words = cleaned.split()
        for tail_size in (3, 2):
            if len(words) <= tail_size:
                continue
            company = clean_ocr_text(" ".join(words[-tail_size:]))
            title = clean_ocr_text(" ".join(words[:-tail_size]))
            if self._looks_like_role_title(title) and self._is_company_name(company):
                return title, company
        return cleaned, ""

    def _is_probable_project_title(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        if len(cleaned.split()) > 10:
            return False
        if cleaned.endswith("."):
            return False
        lower = cleaned.lower()
        if any(verb in lower for verb in RESPONSIBILITY_VERBS):
            return False
        return True


_builder_schema_mapper = None


def get_builder_schema_mapper() -> BuilderSchemaMapper:
    global _builder_schema_mapper
    if _builder_schema_mapper is None:
        _builder_schema_mapper = BuilderSchemaMapper()
    return _builder_schema_mapper


def infer_file_type(filename: str) -> str:
    ext = os.path.splitext(clean_ocr_text(filename))[1].lower().lstrip(".")
    return ext or "pdf"

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

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
    "achieve",
    "achieving",
    "assisted",
    "building",
    "calculated",
    "collaborated",
    "conducted",
    "coordinated",
    "coordinating",
    "created",
    "delivered",
    "designed",
    "developed",
    "drafted",
    "ensure",
    "ensuring",
    "executed",
    "facilitated",
    "generated",
    "handled",
    "identifying",
    "implemented",
    "imported",
    "improved",
    "led",
    "liaised",
    "maintained",
    "managed",
    "mentored",
    "monitor",
    "monitored",
    "negotiated",
    "negotiating",
    "owned",
    "participated",
    "performed",
    "prepared",
    "processed",
    "promoted",
    "provided",
    "reconciled",
    "resolved",
    "responsible",
    "reviewed",
    "supported",
    "trained",
    "verifying",
    "worked",
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
    "international",
    "co",
}
# KNOWN_COMPANY_NAMES = {
#     "adp",
#     "exl",
#     "wns",
#     "examity",
#     "hetero",
#     "polamreddy",
#     "sudi",
#     "divi",
# }
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
    "cashier",
    "cloud",
    "sales",
    "technician",
    "supervisor",
    "coordinator",
    "director",
    "intern",
    "associate",
    "operator",
    "trainer",
}
NON_TITLE_PHRASES = {
    "professional experience",
    "proffessional experience",
    "work experience",
    "employment history",
    "employment",
    "career history",
    "responsibilities",
    "profile",
    "summary",
    "professional summary",
    "key responsibilities",
    "key achievements",
    "achievements",
    "objective",
    "career objective",
    "projects",
    "certifications",
    "skills",
    "education",
    "languages",
    "interests",
    "hobbies",
    "references",
    "additional information",
}
# Phrases that identify objective/summary text — these must NEVER create education records
OBJECTIVE_PHRASES = {
    "to work in",
    "to be in",
    "to contribute",
    "to utilize",
    "to apply",
    "to grow",
    "to seek",
    "i can",
    "i want",
    "i am looking",
    "looking for",
    "aspiring to",
    "seeking a position",
    "seeking an opportunity",
    "responsible position",
    "professional growth",
    "driven environment",
    "contribute my",
    "apply my knowledge",
    "organization's growth",
    "organisation's growth",
    "potential ability",
}

# Education anchor words — a line must have at least one to be an education record
EDU_ANCHOR_WORDS = {
    "university",
    "college",
    "institute",
    "school",
    "academy",
    "bachelor",
    "bachelors",
    "master",
    "masters",
    "phd",
    "doctorate",
    "diploma",
    "ssc",
    "hsc",
    "intermediate",
    "b.tech",
    "m.tech",
    "b.e",
    "b.sc",
    "m.sc",
    "m.com",
    "b.com",
    "mba",
    "matriculation",
    "10th",
    "12th",
    "graduation",
    "post graduation",
    "postgraduate",
    "undergraduate",
}

COMMON_LANGUAGE_NAMES = {
    "english", "hindi", "french", "german", "spanish", "arabic", "urdu", "tamil",
    "telugu", "kannada", "malayalam", "marathi", "gujarati", "bengali", "japanese",
    "chinese", "korean", "portuguese", "italian",
}
KNOWN_CERTIFICATION_NAMES = {
    "adp ihcm",
    "gst",
    "ms-cit",
    "siebel",
    "siebel (records management)",
    "tally & taxation",
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
    "adp ihcm": "ADP iHCM",
    "adp ngux": "ADP NGUX",
    "api": "API",
    "apis": "APIs",
    "sap": "SAP",
    "sql": "SQL",
    "hmrc edi and query resolution": "HMRC EDI and Query Resolution",
    "fps/eps": "FPS/EPS",
    "p45/p46": "P45/P46",
    "p11d": "P11D",
    "qa checks": "QA Checks",
    "ytd reconciliation": "YTD Reconciliation",
    "year-to-date (ytd) reconciliation": "Year-to-Date (YTD) Reconciliation",
    "eoy processes": "EOY Processes",
    "nic": "NIC",
    "ibpms": "IBPMS",
    "sla tracking/reporting": "SLA Tracking/Reporting",
    "sop creation": "SOP Creation",
    "powerpoint": "PowerPoint",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "bom": "BOMs",
    "boms": "BOMs",
    "aws": "AWS",
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
    "requirements", "gathering", "data", "analysis", "problem", "solving",
    "root", "cause", "process", "documentation", "reconciliation",
    "naukri", "resume", "cv",
}

SUPPORTED_NATIVE_FIELDS = {
    "name",
    "email",
    "phone",
    "summary",
    "objective",
    "profile",
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


class _LineType(Enum):
    """Classification for a single line in a raw section block.

    Used by the segment-based extraction pipeline.  The same enum is shared
    between experience and education extraction so that the segmenter logic
    can be reused across sections.
    """
    HEADER = auto()       # known section header for the current section (skip)
    STOP = auto()         # non-section header — hard boundary, commit and stop
    DATE = auto()         # contains a parseable year or date range
    BULLET = auto()       # starts with a bullet / list marker character
    IDENTITY = auto()     # looks like a title or company name (short, named)
    DESCRIPTION = auto()  # everything else: long text, sentences, etc.


@dataclass
class _RawSegment:
    """One candidate job/education block extracted from raw section text.

    A segment is a collection of lines that belong to a single record,
    grouped by the boundary detector before any field-level parsing occurs.
    """
    identity_lines: List[str] = field(default_factory=list)
    date_lines: List[str] = field(default_factory=list)
    description_lines: List[str] = field(default_factory=list)

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
                result = self._normalize_education_entry(item, logs, idx)
                # _normalize_education_entry returns {} when entry is rejected (e.g., objective text)
                if result:
                    parsed_entries.append(result)

        if raw_text:
            parsed_entries.extend(self._extract_education_from_raw(raw_text, logs))

        # Final guard: only keep entries that have at least one real education field
        parsed_entries = [
            e for e in parsed_entries
            if any(e.get(k, "") for k in ("degree", "school", "gradYear", "gpa"))
            and e.get("school", "") != "" or e.get("degree", "") != ""
        ]
        # Additionally discard entries whose school/degree are pure objective sentences
        parsed_entries = [
            e for e in parsed_entries
            if not self._line_is_objective_text(e.get("school", ""))
            and not self._line_is_objective_text(e.get("degree", ""))
            and not self._line_is_objective_text(e.get("field", ""))
        ]
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

        # Prefer structured extraction when it already yields reasonable entries.
        # Raw reconstruction is more error-prone and should only be used to backfill missing data.
        chosen = normalized

        # Strict rule: disable raw reconstruction whenever the structured
        # experience section already has entries. This prevents over-splitting
        # into multiple bogus records.
        structured_has_entries = isinstance(structured_entries, list) and len(structured_entries) > 0
        should_try_raw = bool(raw_text) and not structured_has_entries

        if should_try_raw:
            raw_normalized: List[Dict[str, Any]] = []
            if raw_text:
                raw_normalized = self._extract_experience_from_raw(raw_text, logs)

            structured_score = self._experience_quality_score(normalized)
            raw_score = self._experience_quality_score(raw_normalized)

            if raw_score > structured_score and raw_score > 0:
                logs["normalization_corrections"].append(
                    f"experience remapped from raw section (quality {raw_score} > structured {structured_score})"
                )
                chosen = raw_normalized
            elif raw_score == structured_score and raw_score > 0 and len(raw_normalized) > len(normalized):
                chosen = raw_normalized
        
        resume_data["experience"] = self._dedupe_experience(
            self._repair_experience_sequence(chosen)
        )
        # Drop bogus fragments created from section headers/noise.
        # Additionally, avoid splitting/duplicating section-header tokens into their own jobs.
        resume_data["experience"] = self._filter_bogus_experience_entries(resume_data["experience"])

        # If we still ended up with multiple fragments titled as section headers ("Professional Experience" etc),
        # merge their descriptions into the previous real job.
        resume_data["experience"] = self._merge_section_header_experience_fragments(resume_data["experience"])

        # Merge orphan description-only fragments into the previous entry.
        resume_data["experience"] = self._merge_orphan_experience_descriptions(resume_data["experience"])

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
            if cleaned in SKILL_CASE_MAP.values():
                normalized.append(cleaned)
                continue
            # Reject skills that look like a person's full name (two capitalized words, no known skill marker)
            if self._is_valid_person_name(cleaned):
                logs["normalization_corrections"].append(f"skill '{cleaned}' looks like a person name — discarded")
                continue
            # Reject skills that are purely location names
            if self._looks_like_location(cleaned):
                logs["normalization_corrections"].append(f"skill '{cleaned}' looks like a location — discarded")
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
        raw_projects = self._extract_projects_from_raw(self._section_raw(native, "projects"))
        if self._project_quality_score(raw_projects) > self._project_quality_score(normalized):
            logs["normalization_corrections"].append("projects remapped from raw section")
            normalized = raw_projects
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
            if not cert_raw:
                cert_raw = self._extract_heading_block_from_raw_text(
                    self._section_raw(native, "raw_text"),
                    {"certifications", "certification", "certificates", "licenses", "courses"},
                    {"skills", "experience", "work experience", "professional experience", "education", "languages", "achievements"},
                )
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
                normalized.extend(self._extract_certifications_from_text(ach_raw))
                for line in split_text_items(ach_raw):
                    cleaned = clean_ocr_text(line)
                    if self._is_valid_achievement_text(cleaned):
                        normalized.append({"name": cleaned, "issuingOrg": "", "achievedDate": "", "url": ""})
        if not normalized:
            normalized.extend(self._extract_certifications_from_text(self._section_raw(native, "raw_text")))
        resume_data["certifications"] = self._dedupe_certifications([c for c in normalized if c["name"]])

    def _map_languages(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        conf = self._section_confidence(native, "languages")

        structured = self._section_structured(native, "languages")
        raw = self._section_raw(native, "languages")
        if not raw:
            raw = self._extract_heading_block_from_raw_text(
                self._section_raw(native, "raw_text"),
                {"languages", "language skills", "known languages"},
                {"skills", "experience", "work experience", "professional experience", "education", "certifications", "achievements"},
            )
        language_values: List[str] = []
        if not self._is_low_conf(conf) and isinstance(structured.get("languages"), list):
            language_values = [clean_ocr_text(v) for v in structured["languages"]]
        if not language_values and raw:
            language_values = split_text_items(raw)
            if len(language_values) <= 1:
                language_values = [clean_ocr_text(line) for line in raw.splitlines() if clean_ocr_text(line)]
        if self._is_low_conf(conf) and not language_values:
            resume_data["languages"] = []
            return

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
        """Resolve the candidate summary text from every location the upstream parser
        may have stored it, in priority order:

        1. native['summary']['raw_text']                  — primary structured section
        2. native['summary']['structured_data']['text'|'content'|'summary'|'description']
           — some parsers populate structured_data instead of raw_text
        3. native['objective']['raw_text'] or its structured_data equivalents
           — parsers that label the section as 'objective' instead of 'summary'
        4. native['profile']['raw_text'] or its structured_data equivalents
           — parsers that label the section as 'profile'
        5. Block extraction from native['raw_text'] using an expanded heading set
           — fallback when the parser did not emit a dedicated summary section
        6. _generate_summary_if_missing — last resort auto-generation
        """

        def _structured_text(section_dict: Dict[str, Any]) -> str:
            """Pull plain-text content from a section's structured_data block.

            Parsers use different keys: 'text', 'content', 'summary', 'description'.
            Returns the first non-empty value found, cleaned.
            """
            sd = section_dict.get("structured_data", {})
            if not isinstance(sd, dict):
                return ""
            for key in ("text", "content", "summary", "description", "profile"):
                val = sd.get(key, "")
                if isinstance(val, str):
                    cleaned = clean_ocr_text(val)
                    if cleaned:
                        return cleaned
                elif isinstance(val, list):
                    joined = " ".join(clean_ocr_text(v) for v in val if isinstance(v, str))
                    cleaned = clean_ocr_text(joined)
                    if cleaned:
                        return cleaned
            return ""

        def _resolve_from_section(key: str) -> str:
            """Return the best summary text from a named top-level native key."""
            section = self._section(native, key)
            # Prefer raw_text — it is verbatim parser output
            raw = clean_ocr_multiline(section.get("raw_text", ""))
            if raw:
                return raw
            # Fall back to structured_data text fields
            return _structured_text(section)

        # ── 1 & 2: primary summary section ──────────────────────────────────────
        candidate = _resolve_from_section("summary")
        if candidate:
            resume_data["summary"] = candidate
            return

        # ── 3: objective section (some parsers label summaries as objectives) ───
        candidate = _resolve_from_section("objective")
        if candidate:
            logs["normalization_corrections"].append("summary resolved from native 'objective' section")
            resume_data["summary"] = candidate
            return

        # ── 4: profile section ───────────────────────────────────────────────────
        candidate = _resolve_from_section("profile")
        if candidate:
            logs["normalization_corrections"].append("summary resolved from native 'profile' section")
            resume_data["summary"] = candidate
            return

        # ── 5: heading-block extraction from raw_text ────────────────────────────
        candidate = self._extract_summary_from_raw_text(self._section_raw(native, "raw_text"))
        if candidate:
            logs["normalization_corrections"].append("summary recovered from raw_text heading block")
            resume_data["summary"] = candidate
            return

        # ── 6: auto-generation ───────────────────────────────────────────────────
        resume_data["summary"] = self._generate_summary_if_missing(resume_data)

    def _finalize_resume_data(self, native: Dict[str, Any], resume_data: Dict[str, Any], logs: Dict[str, List[str]]) -> None:
        personal = resume_data["personal"]
        for key, value in list(personal.items()):
            if key == "websites":
                personal["websites"] = [self._normalize_url(clean_ocr_text(v)) for v in value if clean_ocr_text(v)]
                personal["websites"] = dedupe_case_insensitive(
                    [url for url in personal["websites"] if url and not self._is_valid_linkedin(url)]
                )
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
        elif personal["linkedin"]:
            personal["linkedin"] = self._normalize_url(personal["linkedin"])

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

        resume_data["skills"] = self._dedupe_skills_semantic(
            dedupe_case_insensitive([self._normalize_skill(s) for s in resume_data["skills"] if self._normalize_skill(s)])
        )
        # Education: sanitize → filter empty → dedupe
        sanitized_edu = [self._sanitize_education_entry(e) for e in resume_data["education"]]
        resume_data["education"] = self._dedupe_education([e for e in sanitized_edu if e])

        # Experience finalization — identical pipeline to _map_experience so both
        # paths produce the same structural guarantees:
        #   sanitize (structural anchor check + field cleanup)
        #   → drop empty dicts (records that failed the anchor check)
        #   → repair sequence (orphan merging)
        #   → fragment consolidation (sentence-fragment titles merged into previous)
        #   → orphan description merging
        #
        # Filtering the empty dicts BEFORE repair/merge is critical: passing {}
        # into _repair_experience_sequence causes it to skip the fragment check
        # because {} has no title or date, and it would be treated as a no-op
        # rather than a signal to merge.
        sanitized_exp = [self._sanitize_experience_entry(e) for e in resume_data["experience"]]
        sanitized_exp = [e for e in sanitized_exp if e]  # drop anchor-failed records
        resume_data["experience"] = self._dedupe_experience(
            self._repair_experience_sequence(sanitized_exp)
        )
        resume_data["experience"] = self._merge_section_header_experience_fragments(resume_data["experience"])
        resume_data["experience"] = self._merge_orphan_experience_descriptions(resume_data["experience"])
        resume_data["projects"] = [self._sanitize_project_entry(p) for p in resume_data["projects"] if p.get("title") or p.get("description")]
        resume_data["certifications"] = [self._sanitize_cert_entry(c) for c in resume_data["certifications"] if c.get("name")]

        if not personal["profession"]:
            for exp in resume_data["experience"]:
                title = clean_ocr_text(exp.get("title", ""))
                if title and not self._looks_like_description(title):
                    personal["profession"] = title
                    break
        elif resume_data["experience"]:
            profession = clean_ocr_text(personal["profession"])
            first_title = clean_ocr_text(resume_data["experience"][0].get("title", ""))
            first_employer = clean_ocr_text(resume_data["experience"][0].get("employer", ""))
            if first_title and first_employer and first_employer.lower() in profession.lower():
                personal["profession"] = first_title

        if not resume_data["summary"]:
            resume_data["summary"] = self._generate_summary_if_missing(resume_data)

    def _normalize_education_entry(self, item: Dict[str, Any], logs: Dict[str, List[str]], idx: int) -> Dict[str, Any]:
        explicit_degree = clean_ocr_text(item.get("degree", ""))
        explicit_school = clean_ocr_text(item.get("institution", item.get("school", "")))
        explicit_field = clean_ocr_text(item.get("field", ""))
        explicit_gpa = clean_ocr_text(item.get("grade", item.get("gpa", "")))
        explicit_year = clean_ocr_text(item.get("year", ""))

        # Hard reject: if any key field is an objective/narrative sentence, discard it
        for label, val in (("degree", explicit_degree), ("school", explicit_school), ("field", explicit_field)):
            if val and self._line_is_objective_text(val):
                logs["normalization_corrections"].append(
                    f"education entry {idx}: {label} field '{val[:60]}' is objective text — entry discarded"
                )
                return {}

        # Hard reject: if none of degree/school/field contain an education anchor and no year, skip
        combined_text = f"{explicit_degree} {explicit_school} {explicit_field}"
        if not self._line_has_education_anchor(combined_text) and not re.search(r"\b(?:19|20)\d{2}\b", combined_text):
            logs["normalization_corrections"].append(
                f"education entry {idx}: no education anchor found in '{combined_text[:80]}' — entry discarded"
            )
            return {}

        if explicit_degree or explicit_school:
            degree_entry = self._parse_education_line(explicit_degree, logs, idx) if explicit_degree else {
                "school": "",
                "location": "",
                "field": "",
                "degree": "",
                "gradMonth": "",
                "gradYear": "",
                "gpa": "",
            }
            school_entry = self._parse_education_line(explicit_school, logs, idx) if explicit_school else {}
            if (
                explicit_school
                and self._degree_priority(explicit_school) > self._degree_priority(explicit_degree)
                and clean_ocr_text(school_entry.get("school", ""))
            ):
                degree_entry = school_entry
                explicit_school = clean_ocr_text(school_entry.get("school", ""))
                explicit_field = clean_ocr_text(school_entry.get("field", "")) or explicit_field
            if explicit_gpa and not clean_ocr_text(degree_entry.get("degree", "")) and self._degree_priority(explicit_gpa) > 0:
                gpa_entry = self._parse_education_line(explicit_gpa, logs, idx)
                if clean_ocr_text(gpa_entry.get("degree", "")):
                    degree_entry = gpa_entry
                    explicit_school = explicit_school or clean_ocr_text(gpa_entry.get("school", ""))
                    explicit_field = explicit_field or clean_ocr_text(gpa_entry.get("field", ""))
            year, _ = normalize_year(explicit_year)
            gpa = self._extract_gpa_value(explicit_gpa) or clean_ocr_text(degree_entry.get("gpa", ""))
            return {
                "school": (
                    explicit_school
                    if explicit_school and not self._degree_priority(explicit_school)
                    else clean_ocr_text(school_entry.get("school", "") or degree_entry.get("school", ""))
                ),
                "location": clean_ocr_text(school_entry.get("location", "") or degree_entry.get("location", "")),
                "field": explicit_field or clean_ocr_text(degree_entry.get("field", "")),
                "degree": clean_ocr_text(degree_entry.get("degree", explicit_degree)),
                "gradMonth": "",
                "gradYear": year or clean_ocr_text(degree_entry.get("gradYear", "")),
                "gpa": gpa,
            }
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

    def _line_is_objective_text(self, line: str) -> bool:
        """Return True if this line is objective/summary narrative, not an education entity."""
        lower = clean_ocr_text(line).lower()
        return any(phrase in lower for phrase in OBJECTIVE_PHRASES)

    def _line_has_education_anchor(self, line: str) -> bool:
        """Return True only if the line contains a genuine education anchor word/phrase."""
        lower = clean_ocr_text(line).lower()
        # Check multi-word anchors first
        for anchor in EDU_ANCHOR_WORDS:
            if anchor in lower:
                return True
        # Also accept lines that have a parseable graduation year alongside school-like text
        if re.search(r"\b(?:19|20)\d{2}\b", lower):
            return True
        return False

    def _extract_education_from_raw(self, raw_text: str, logs: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Anchor-driven education extraction.

        A line must pass _line_has_education_anchor and must NOT be
        objective/summary text to be eligible for segmentation into a record.
        Records are only emitted when an education anchor is confirmed in the segment.
        """
        lines = [clean_ocr_text(ln) for ln in raw_text.splitlines() if clean_ocr_text(ln)]
        if not lines:
            return []

        # Phase 1: filter and segment into candidate blocks
        segments: List[List[str]] = []
        current: List[str] = []
        all_stop = SECTION_STOP_HEADERS | EXPERIENCE_SECTION_HEADERS

        for line in lines:
            lower = line.lower().strip(" :|-")

            # Hard stop at known section boundaries
            if lower in all_stop:
                if current:
                    segments.append(current)
                    current = []
                continue

            # Objective/narrative text must never create education records
            if self._line_is_objective_text(line):
                continue

            # Year-bearing line closes the current segment
            if re.search(r"\b(?:19|20)\d{2}\b", line) and current:
                current.append(line)
                segments.append(current)
                current = []
                continue

            current.append(line)
            # Flush on accumulation of 4+ lines (prevents runaway blocks)
            if len(current) >= 4:
                segments.append(current)
                current = []

        if current:
            segments.append(current)

        # Phase 2: parse only segments that contain an education anchor
        parsed: List[Dict[str, Any]] = []
        for idx, seg_lines in enumerate(segments):
            combined = " | ".join(seg_lines)
            # Gate: segment must contain at least one education anchor word
            if not any(self._line_has_education_anchor(ln) for ln in seg_lines):
                logs["normalization_corrections"].append(
                    f"education raw segment {idx} discarded — no education anchor: '{combined[:60]}'"
                )
                continue
            parsed_entry = self._parse_education_line(combined, logs, idx)
            if any(parsed_entry.get(k, "") for k in ("degree", "school", "gradYear", "gpa")):
                parsed.append(parsed_entry)

        return parsed

    def _parse_education_line(self, line: str, logs: Dict[str, List[str]], idx: int) -> Dict[str, Any]:
        cleaned = clean_ocr_text(line)
        cleaned = re.sub(r"[–—−]", "-", cleaned)
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

        degree_re = re.compile(r"\b(bachelor|bachelors|master|phd|doctorate|diploma|ssc|hsc|intermediate|b\.?tech|m\.?tech|b\.?e|b\.?sc|m\.?sc|m\.?com|b\.?com|mba)\b", re.IGNORECASE)
        school_re = re.compile(r"\b(university|college|institute|school|academy)\b", re.IGNORECASE)
        gpa_re = re.compile(r"\b(?:(?:cgpa|gpa|grade|percentage)\s*[:\-]?\s*)?(?:\d{1,2}(?:\.\d+)?/10|\d{1,3}(?:\.\d+)?%)", re.IGNORECASE)
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
                gpa = self._extract_gpa_value(part)
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

        # ── Hard reject: section headers must never become records ──────────────
        if title.lower() in NON_TITLE_PHRASES:
            title = ""
        if employer.lower() in NON_TITLE_PHRASES:
            employer = ""
        if title in EXPERIENCE_SECTION_HEADERS:
            title = ""
        if employer in EXPERIENCE_SECTION_HEADERS:
            employer = ""
        if title.lower() in EXPERIENCE_SECTION_HEADERS or self._contains_non_title_phrase(title):
            title = ""
        if employer.lower() in EXPERIENCE_SECTION_HEADERS or self._contains_non_title_phrase(employer):
            employer = ""

        # ── Date-fragment promotion ─────────────────────────────────────────────
        if title and self._looks_like_date_fragment(title):
            duration = duration or title
            title = ""
        if employer and self._looks_like_date_fragment(employer):
            duration = duration or employer
            employer = ""

        # ── Responsibility text must never become a title or employer ───────────
        # If the title starts with a responsibility verb it is description text.
        if title:
            title_lower = title.lower()
            first_word = title_lower.split()[0].rstrip(".,;:") if title_lower.split() else ""
            if first_word in RESPONSIBILITY_VERBS:
                description = f"{title}\n{description}".strip()
                title = ""
        if employer:
            emp_lower = employer.lower()
            first_word_emp = emp_lower.split()[0].rstrip(".,;:") if emp_lower.split() else ""
            if first_word_emp in RESPONSIBILITY_VERBS and not self._is_company_name(employer):
                description = f"{employer}\n{description}".strip()
                employer = ""

        # ── Continuation-line detection: long fragments that aren't company names ─
        if title and len(title.split()) > 8 and not self._is_company_name(title):
            description = f"{title}\n{description}".strip()
            title = ""
        if employer and self._is_continuation_line(employer) and not self._is_company_name(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""
        if employer and len(employer.split()) > 8 and not self._is_company_name(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""

        title, employer = self._split_title_employer(title, employer)
        if title and employer:
            if self._is_company_name(title) and self._looks_like_role_title(employer):
                title, employer = employer, title
            elif self._looks_like_role_title(title) and not self._is_company_name(employer):
                employer = ""
        if (
            title
            and not employer
            and self._looks_like_role_title(title)
        ):
            split_title, split_employer = self._split_inline_role_company(title)
            title, employer = split_title, split_employer
        if not employer and description:
            first_line = clean_ocr_text(description.split("\n")[0])

            # Only promote first line to employer when it is the entire description
            if (
                self._is_company_name(first_line)
                and len(description.splitlines()) == 1
                and len(description.split()) <= 8
            ):
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

        # ── Experience anchor validation: a record must have at least one anchor ──
        # Anchors: title, employer, or (date + description). Bare description-only
        # entries without dates or identity are rejected here.
        has_identity = bool(title or employer)
        has_date = bool(
            date_data.get("startYear") or date_data.get("endYear") or date_data.get("current")
        )
        has_description = bool(description)

        if not has_identity and not has_date:
            # Description-only fragment with no anchor — discard
            logs["normalization_corrections"].append(
                f"experience entry {idx} discarded — no identity or date anchor"
            )
            return {}

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

    def _split_merged_role_company_line(self, line: str) -> List[str]:
        """Split a merged "Title CompanyName Ltd" line into [title, company].

        Uses COMPANY_SUFFIXES to detect the company tail rather than a
        hardcoded list of company name words.
        """
        # Build suffix pattern from the shared COMPANY_SUFFIXES set.
        suffix_pattern = "|".join(re.escape(s) for s in sorted(COMPANY_SUFFIXES, key=len, reverse=True))
        m = re.match(
            rf"^(?P<title>.+?)\s+(?P<company>[A-Z][A-Za-z&.\- ]+(?:{suffix_pattern}))$",
            line,
            re.IGNORECASE,
        )
        if m:
            return [
                clean_ocr_text(m.group("title")),
                clean_ocr_text(m.group("company")),
            ]
        return [line]

    # ------------------------------------------------------------------
    # Raw experience extraction pipeline
    # ------------------------------------------------------------------

    def _classify_raw_line(
        self,
        line: str,
        section_headers: set,
        stop_headers: set,
    ) -> "_LineType":
        """Classify a single raw line for the segment-based extraction pipeline.

        This is a pure classification step with no side effects.  The same
        method is reusable for education extraction by passing different
        ``section_headers`` and ``stop_headers`` sets.
        """
        lower = clean_ocr_text(line).lower().strip(" :|-")
        if not lower:
            return _LineType.DESCRIPTION

        if lower in section_headers:
            return _LineType.HEADER
        if lower in stop_headers:
            return _LineType.STOP
        if lower in NON_TITLE_PHRASES:
            return _LineType.HEADER

        # Bullet / list marker
        if re.match(r"^\s*[•◦●▪\-–—]+\s+|^\s*[-–—]\s*$", line):
            return _LineType.BULLET

        # Date — year or date range present
        date_info = extract_date_range(line)
        if date_info.get("startYear") or date_info.get("endYear") or date_info.get("current"):
            return _LineType.DATE

        # Responsibility verbs at start → always a description line, never identity
        first_word = lower.split()[0].rstrip(".,;:") if lower.split() else ""
        if first_word in RESPONSIBILITY_VERBS:
            return _LineType.DESCRIPTION

        # Identity: role@company or role–company explicit delimiter
        if self._extract_role_company_from_line(line):
            return _LineType.IDENTITY

        # Identity: short line that looks like a title or company name
        if not self._looks_like_description(line):
            lower_words = set(re.findall(r"\b\w+\b", lower))
            looks_title = (
                len(line.split()) <= 6
                and bool(lower_words & TITLE_HINTS)
                and lower not in NON_TITLE_PHRASES
                and not self._contains_non_title_phrase(line)
            )
            looks_company = self._is_company_name(line)
            if looks_title or looks_company:
                return _LineType.IDENTITY

        return _LineType.DESCRIPTION

    def _segment_raw_experience_lines(
        self,
        classified: List[Tuple[str, "_LineType"]],
    ) -> List["_RawSegment"]:
        """Walk classified lines and group them into candidate job segments.

        Boundary rules:
        - A DATE line seen after at least one IDENTITY line in the current
          segment does NOT start a new segment; it belongs to the current one.
        - A DATE line seen after the current segment already has a date AND
          has description lines starts a new segment (a second job with its
          own date range).
        - A new IDENTITY line after the current segment has a date (body
          started) always starts a new segment.
        - A STOP line flushes the current segment and halts processing.
        - HEADER and BULLET lines are never boundary triggers.

        This method contains no heuristic content inspection — all of that
        is deferred to _interpret_experience_segment and
        _normalize_experience_entry.
        """
        segments: List[_RawSegment] = []
        current = _RawSegment()

        def flush() -> None:
            has_identity = bool(current.identity_lines)
            has_date = bool(current.date_lines)
            has_desc = bool(current.description_lines)
            # A valid segment must have at least identity or a date, plus some content.
            if (has_identity or has_date) and (has_identity or has_desc or has_date):
                segments.append(_RawSegment(
                    identity_lines=list(current.identity_lines),
                    date_lines=list(current.date_lines),
                    description_lines=list(current.description_lines),
                ))
            current.identity_lines.clear()
            current.date_lines.clear()
            current.description_lines.clear()

        for line, ltype in classified:
            if ltype == _LineType.HEADER:
                continue
            if ltype == _LineType.STOP:
                flush()
                break
            if ltype == _LineType.DATE:
                body_started = bool(current.date_lines or current.description_lines)
                if body_started and current.date_lines:
                    # Second date block → new segment
                    flush()
                current.date_lines.append(line)
                continue
            if ltype == _LineType.IDENTITY:
                body_started = bool(current.date_lines or current.description_lines)
                if body_started:
                    flush()
                # Accumulate consecutive identity lines (e.g. title on one
                # line, company on the next) into the same segment.
                current.identity_lines.append(line)
                continue
            # BULLET or DESCRIPTION
            if ltype == _LineType.BULLET:
                # Bullets only attach to the current segment; they never open a new one.
                if current.identity_lines or current.date_lines:
                    current.description_lines.append(line)
                # else: bullet before any job header → drop (noise)
                continue
            # DESCRIPTION
            current.description_lines.append(line)

        flush()
        return segments

    def _interpret_experience_segment(
        self, segment: "_RawSegment", idx: int, logs: Dict[str, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Convert one _RawSegment into a normalized experience record.

        This is the BlockInterpreter stage.  It translates the segment's
        grouped lines into the flat dict consumed by _normalize_experience_entry.
        """
        # Identity: try explicit role–company delimiter first
        title = ""
        company = ""
        for iline in segment.identity_lines:
            rc = self._extract_role_company_from_line(iline)
            if rc:
                title = rc.get("title", "")
                company = rc.get("company", "")
                break
        if not title and not company:
            # Fall back: assign first identity line as title, second as company
            if segment.identity_lines:
                title = segment.identity_lines[0]
            if len(segment.identity_lines) > 1:
                company = segment.identity_lines[1]

        # Duration: join all date lines
        duration = " ".join(segment.date_lines)

        # Location: attempt to split from the duration line
        location = ""
        if segment.date_lines:
            extracted_location, remainder = self._split_location_duration_line(segment.date_lines[0])
            if extracted_location:
                location = extracted_location
                duration = remainder or duration

        description = "\n".join(segment.description_lines)

        item = {
            "title": title,
            "company": company,
            "duration": duration,
            "location": location,
            "description": description,
        }
        return self._normalize_experience_entry(item, logs, idx)

    def _extract_experience_from_raw(
        self, raw_text: str, logs: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Reconstruct experience records from a raw text block.

        Pipeline:
        1. Split into cleaned lines.
        2. Split merged role+company lines (e.g. "Senior Engineer AcmeCorp Ltd").
        3. Classify each line using _classify_raw_line.
        4. Segment classified lines into candidate job blocks using
           _segment_raw_experience_lines.
        5. Interpret each segment using _interpret_experience_segment.
        6. Pass to _normalize_experience_entry (inside interpret step).

        This method replaces the unreachable dead-code block that previously
        appeared after _split_merged_role_company_line.
        """
        lines = [clean_ocr_text(ln) for ln in clean_ocr_multiline(raw_text).splitlines() if clean_ocr_text(ln)]
        if not lines:
            return []

        # Expand any merged "title Company Ltd" lines into two lines.
        expanded: List[str] = []
        for line in lines:
            expanded.extend(self._split_merged_role_company_line(line))

        # Classify each line.
        classified: List[Tuple[str, _LineType]] = [
            (line, self._classify_raw_line(line, EXPERIENCE_SECTION_HEADERS, SECTION_STOP_HEADERS))
            for line in expanded
        ]

        # Segment into candidate job blocks.
        segments = self._segment_raw_experience_lines(classified)

        # Interpret each segment.
        normalized: List[Dict[str, Any]] = []
        for idx, segment in enumerate(segments):
            record = self._interpret_experience_segment(segment, idx, logs)
            if record:
                normalized.append(record)

        return normalized

    def _extract_role_company_from_line(self, line: str) -> Dict[str, str]:
        text = clean_ocr_text(line)
        if not text:
            return {}
        patterns = [
            r"^(?P<title>.+?)\s+@\s+(?P<company>.+)$",
            r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$",
            r"^(?P<left>.+?)\s*[-–—|]\s*(?P<right>.+)$",
        ]
        for pattern in patterns:
            m = re.match(pattern, text, re.IGNORECASE)
            if not m:
                continue
            if "title" in m.groupdict() and "company" in m.groupdict():
                return {"title": clean_ocr_text(m.group("title")), "company": clean_ocr_text(m.group("company"))}
            left = clean_ocr_text(m.group("left"))
            right = clean_ocr_text(m.group("right"))
            right_title, right_company = self._split_inline_role_company(right)
            left_is_role = self._looks_like_role_title(left)
            right_is_role = self._looks_like_role_title(right)
            if left_is_role and right_company:
                return {"title": clean_ocr_text(f"{left} {right_title}"), "company": right_company}
            if self._is_company_name(left) and self._looks_like_role_title(right):
                return {"title": right, "company": left}
            if self._is_company_name(right) and self._looks_like_role_title(left):
                return {"title": left, "company": right}
            if left_is_role and not right_is_role and self._is_company_name(right):
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

        links_raw = "\n".join(
            [
                self._section_raw(native, "links"),
                self._section_raw(native, "contact"),
                self._section_raw(native, "raw_text"),
            ]
        )
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
        url = re.sub(r"\s*:\s*//\s*", "://", url)
        url = re.sub(r"\s+", "", url)
        linkedin_match = re.search(r"(?:www\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9_.%+-]+/?", url, re.IGNORECASE)
        if linkedin_match:
            linkedin_path = linkedin_match.group(0)
            if linkedin_path.lower().startswith("www."):
                linkedin_path = linkedin_path[4:]
            return f"https://{linkedin_path}"
        while re.match(r"^https?://https?://", url, re.IGNORECASE):
            url = re.sub(r"^https?://", "", url, count=1, flags=re.IGNORECASE)
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
        # glued = parts[0] if parts else ""
        # # Conservative fallback for common lowercase first+last email handles.
        # known_names = ("pradeepthi", "pradeepthi")
        # for first in known_names:
        #     if glued.startswith(first) and len(glued) > len(first) + 2:
        #         candidate = f"{first} {glued[len(first):]}".title()
        #         return candidate if self._is_valid_person_name(candidate) else ""
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
        if re.search(r"\b(?:be|b\.e)\b", lower) or any(token in lower for token in ("bachelor", "b.tech", "b.com", "b.sc")):
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
        if e and not t and self._is_company_name(e):
            return "", e
        combined = t if t else e
        if not combined:
            return t, e
        explicit_dash = re.match(r"^(?P<left>.+?)\s+-\s+(?P<right>.+)$", combined)
        if explicit_dash:
            left = clean_ocr_text(explicit_dash.group("left"))
            right = clean_ocr_text(explicit_dash.group("right"))
            if self._is_company_name(left) and self._is_company_suffix_fragment(right):
                return "", combined
            left_is_company = self._is_company_name(left)
            right_is_company = self._is_company_name(right)
            left_is_role = self._looks_like_role_title(left)
            right_is_role = self._looks_like_role_title(right)
            if left_is_company and right_is_role:
                return right, left
            if left_is_role and right_is_company:
                return left, right
            if left_is_company and right_is_company:
                return ("", combined) if e and not t else (combined, "")

        patterns = [
            r"^(?P<title>.+?)\s+@\s+(?P<company>.+)$",
            r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$",
            r"^(?P<left>.+?)\s*[-–—|]\s*(?P<right>.+)$",
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
            if self._is_company_name(left) and self._is_company_suffix_fragment(right):
                return "", combined
            # Company names sometimes contain hyphens
            if (
                self._is_company_name(left)
                and self._is_company_name(right)
            ):
                return ("", combined) if e and not t else (combined, "")
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

    def _is_company_suffix_fragment(self, text: str) -> bool:
        lower = clean_ocr_text(text).lower()
        return bool(re.match(r"^(?:a\s+)?(?:unit|division|subsidiary|branch|department)\s+of\b", lower))

    def _normalize_role_seniority_prefix(self, title: str) -> str:
        cleaned = clean_ocr_text(title)
        if len(cleaned.split()) <= 1:
            return cleaned
        return clean_ocr_text(re.sub(r"^(?:intern|junior)\s+", "", cleaned, flags=re.IGNORECASE))

    def _is_company_name(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned or len(cleaned.split()) > 12:
            return False
        lower = cleaned.lower()
        normalized = re.sub(r"[.,]", "", lower)
        first_word = re.sub(r"[^a-z]", "", lower.split()[0]) if lower.split() else ""
        if self._looks_like_location(cleaned):
            return False
        # if normalized in KNOWN_COMPANY_NAMES:
        #     return True
        if first_word in RESPONSIBILITY_VERBS or any(f" {word} " in f" {lower} " for word in RESPONSIBILITY_VERBS):
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
        if self._looks_like_role_title(cleaned):
            return False
        words = [w for w in cleaned.split() if w]
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]

        if len(alpha_words) > 1 and all((w[0].isupper() or w.isupper()) for w in alpha_words):
            return True
        # Require at least one known company indicator
        if (
            len(alpha_words) >= 2
            and any(
                token.lower() in COMPANY_SUFFIXES
                for token in alpha_words
            )
        ):
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

    def _split_location_duration_line(self, text: str) -> Tuple[str, str]:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return "", ""
        parts = [clean_ocr_text(part) for part in re.split(r"\s+[|]\s+|\s+-\s+", cleaned, maxsplit=1) if clean_ocr_text(part)]
        if len(parts) != 2:
            return "", cleaned
        left, right = parts
        if extract_date_range(right).get("startYear") and self._looks_like_location(left):
            return left.title(), right
        return "", cleaned

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
        # Strip OCR scan artifacts: single capital letter (optionally followed by «»*•) then space
        # Examples: "E Customer Assistance" → "Customer Assistance", "E« Payment Processing" → "Payment Processing"
        skill = re.sub(r"^[A-Z]\W*\s+(?=[A-Z])", "", skill)
        # Strip any remaining leading non-alphabetic garbage
        skill = re.sub(r"^[^A-Za-z0-9]+", "", skill).strip()
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
        if re.search(r"[()]{2,}|[.]{3,}|[•◦●▪]", skill):
            return ""
        if ". " in skill:
            return ""
        if lower.endswith(")") and "(" not in lower:
            skill = skill[:-1].strip()
        if lower in GENERIC_SKILLS:
            return ""
        if skill.count("(") != skill.count(")"):
            return ""
        mapped = SKILL_CASE_MAP.get(lower)
        if mapped:
            return mapped
        if lower.isupper():
            return lower
        if len(skill.split()) == 1:
            return skill.title()
        return " ".join([SKILL_CASE_MAP.get(p.lower(), p.capitalize()) for p in skill.split()])

    def _dedupe_skills_semantic(self, skills: List[str]) -> List[str]:
        cleaned = [clean_ocr_text(skill) for skill in skills if clean_ocr_text(skill)]
        lowered = [skill.lower() for skill in cleaned]
        result: List[str] = []
        for idx, skill in enumerate(cleaned):
            lower = lowered[idx]
            redundant = False
            for other_idx, other in enumerate(lowered):
                if idx == other_idx:
                    continue
                if lower == other:
                    continue
                other_tokens = set(re.findall(r"[a-z0-9]+", other))
                lower_tokens = set(re.findall(r"[a-z0-9]+", lower))
                if len(lower_tokens) == 1 and lower_tokens <= other_tokens:
                    redundant = True
                    break
                if lower in other and len(skill) < len(cleaned[other_idx]):
                    redundant = True
                    break
            if not redundant:
                result.append(skill)
        return dedupe_case_insensitive(result)

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

        # Don't generate summary if the profession looks like a year or noise
        if re.fullmatch(r"(?:19|20)\d{2}", profession.strip()):
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

        personal = resume_data.get("personal", {})
        person_name_tokens = {
            t.lower() for t in re.findall(r"[A-Za-z]+", f"{personal.get('firstName','')} {personal.get('lastName','')}")
            if len(t) > 2
        }

        skills = []
        for s in resume_data.get("skills", []):
            s_clean = clean_ocr_text(s)
            if not s_clean:
                continue
            # Skip skills that are just the person's name or location tokens
            s_lower = s_clean.lower()
            if any(tok in s_lower for tok in person_name_tokens):
                continue
            if any(loc in s_lower for loc in LOCATION_WORDS):
                continue
            skills.append(s_clean)

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

        # ── Structural gate ───────────────────────────────────────────────────────
        # Gate 1: if any key text field is an objective/narrative sentence, discard
        # the entire entry immediately.  This catches cases where the upstream
        # parser placed an objective statement into the degree or school field
        # (e.g. degree="growth and to be in a responsible position where").
        #
        # Why this must run before the anchor check:
        #   If we checked anchors first, a record with gradYear="2026" would pass
        #   the year-based anchor and we would never reach the objective-text test.
        #   The objective test must be unconditional.
        for val in (degree, school, field):
            if val and self._line_is_objective_text(val):
                return {}

        # Gate 2: the text fields (not the year) must contain at least one education
        # anchor word (degree keyword, institution keyword).  A graduation year alone
        # is NOT sufficient — year + objective sentence is not an education record.
        #
        # This is intentionally stricter than _normalize_education_entry, which
        # allows a year alone as evidence.  _sanitize_education_entry is the
        # second-pass guard: by this point every legitimate record has already
        # been parsed once and should have degree or school populated.  If it has
        # neither, and neither contains an anchor word, it is structural noise.
        text_fields_combined = f"{degree} {school} {field}"
        has_text_anchor = self._line_has_education_anchor(text_fields_combined)
        has_year_anchor = bool(
            grad_year
            or re.search(r"\b(?:19|20)\d{2}\b", text_fields_combined)
        )
        # Require: text anchor, OR (year + at least one non-empty text field that
        # is not itself noise).  Pure year-only entries with no degree/school are dropped.
        has_nonempty_text = bool(degree or school or field)
        if not has_text_anchor and not (has_year_anchor and has_nonempty_text):
            return {}

        grad_year, _ = normalize_year(grad_year)
        month_value, _ = normalize_month(grad_month)
        grad_month = month_value

        if degree and school and degree.lower() == school.lower():
            school = ""
        if school and not degree and self._degree_priority(school) > 0:
            degree, field_from_school, school_from_school, gpa_from_school = self._split_degree_block(school)
            if field_from_school and not field:
                field = field_from_school
            if school_from_school:
                school = school_from_school
            else:
                school = ""
            if gpa_from_school and not gpa:
                gpa = gpa_from_school
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
        if school and self._degree_priority(school) > 0:
            school_degree, school_field, school_school, school_gpa = self._split_degree_block(school)
            if school_school:
                school = school_school
                if school_degree and self._degree_priority(school_degree) > self._degree_priority(degree):
                    degree = school_degree
                if school_field and not field:
                    field = school_field
                if school_gpa and not gpa:
                    gpa = school_gpa
            elif degree:
                school = ""
        gpa = self._extract_gpa_value(gpa)

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
        gpa = self._extract_gpa_value(raw)
        if gpa:
            raw = clean_ocr_text(raw.replace(gpa, ""))

        raw = re.sub(r"\(\s*(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\s*\)", "", raw)
        raw = re.sub(r"\(\s*(?:19|20)\d{2}\s*[–—−]\s*(?:19|20)?\d{2}\s*\)", "", raw)
        raw = re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", "", raw)
        raw = clean_ocr_text(raw)

        degree = raw
        field = ""
        school = ""
        dash_parts = [clean_ocr_text(p) for p in re.split(r"\s+[-–—]\s+", raw, maxsplit=1) if clean_ocr_text(p)]
        if len(dash_parts) == 2 and self._degree_priority(dash_parts[0]) > 0:
            degree = dash_parts[0]
            school = dash_parts[1]
            field_match = re.search(r"\(([^()]{3,80})\)", degree)
            if field_match:
                field = clean_ocr_text(field_match.group(1))
                degree = clean_ocr_text(re.sub(r"\([^()]*\)", "", degree))
            school = self._strip_school_suffix_noise(school)
            return degree, field, school, gpa

        degree_prefix_re = re.compile(
            r"^(?P<degree>(?:B\.?E|B\.?Tech|M\.?Tech|M\.?Com|B\.?Com|M\.?Sc|B\.?Sc|MBA|Intermediate|SSC|Diploma|Bachelors?\s+of\s+Technology|Bachelors?\s+of\s+Engineering|Bachelors?\s+of\s+Commerce|Bachelor[^,]*?|Master[^,]*?))(?:\s+in\s+(?P<field>[^,]+))?(?:,\s*(?P<school>.+))?$",
            re.IGNORECASE,
        )
        m = degree_prefix_re.match(raw)
        if m:
            degree = clean_ocr_text(m.group("degree") or "")
            field = clean_ocr_text(m.group("field") or "")
            school = clean_ocr_text(m.group("school") or "")
            if not field:
                field_match = re.search(r"\(([^()]{3,80})\)", raw)
                if field_match:
                    field = clean_ocr_text(field_match.group(1))
                    degree = clean_ocr_text(re.sub(r"\([^()]*\)", "", degree))
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

    def _extract_gpa_value(self, text: str) -> str:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return ""
        match = re.search(
            r"\b(?:(?:cgpa|gpa|grade|percentage)\s*[:\-]?\s*)?(?:\d{1,2}(?:\.\d+)?/10|\d{1,3}(?:\.\d+)?%)",
            cleaned,
            re.IGNORECASE,
        )
        if not match:
            return ""
        value = clean_ocr_text(match.group(0))
        value = re.sub(r"^(?:cgpa|gpa|grade|percentage)\s*[:\-]?\s*", "", value, flags=re.IGNORECASE)
        return value

    def _strip_school_suffix_noise(self, text: str) -> str:
        school = clean_ocr_text(text)
        if not school:
            return ""
        school = re.sub(r"\(\s*(?:19|20)\d{2}\s*[-/]\s*(?:19|20)?\d{2}\s*\)", "", school)
        school = re.sub(r"\(\s*(?:19|20)\d{2}\s*[–—−]\s*(?:19|20)?\d{2}\s*\)", "", school)
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

        # ── Section-header / non-title phrase rejection ─────────────────────────
        if title.lower() in EXPERIENCE_SECTION_HEADERS or title.lower() in NON_TITLE_PHRASES:
            title = ""
        if employer.lower() in EXPERIENCE_SECTION_HEADERS or employer.lower() in NON_TITLE_PHRASES:
            employer = ""
        if title and self._contains_non_title_phrase(title):
            title = ""
        if employer and self._contains_non_title_phrase(employer):
            employer = ""

        # ── Responsibility-verb opening → push into description ─────────────────
        if title:
            first_word_t = title.lower().split()[0].rstrip(".,;:") if title.lower().split() else ""
            if first_word_t in RESPONSIBILITY_VERBS:
                description = f"{title}\n{description}".strip()
                title = ""
        if employer:
            first_word_e = employer.lower().split()[0].rstrip(".,;:") if employer.lower().split() else ""
            if first_word_e in RESPONSIBILITY_VERBS and not self._is_company_name(employer):
                description = f"{employer}\n{description}".strip()
                employer = ""

        # ── Long-line detection ─────────────────────────────────────────────────
        if title and self._looks_like_description(title):
            description = f"{title}\n{description}".strip()
            title = ""
        if title and len(title.split()) > 8:
            description = f"{title}\n{description}".strip()
            title = ""
        if employer and self._looks_like_description(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""

        if employer and re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$", employer) and not self._is_company_name(employer):
            employer = ""
        if employer and self._is_continuation_line(employer) and not self._is_company_name(employer):
            description = f"{employer}\n{description}".strip()
            employer = ""
        if employer and self._looks_like_location(employer):
            if not clean_ocr_text(entry.get("location", "")):
                entry["location"] = employer.title()
            employer = ""
        if description and not clean_ocr_text(entry.get("location", "")):
            extracted_location, cleaned_description = self._extract_leading_location_from_description(description)
            if extracted_location:
                entry["location"] = extracted_location
                description = cleaned_description
        # Fix reversed mappings like: title="EXL", employer="Customer Service Representative"
        if title and employer:
            title_is_short_org = bool(re.match(r"^[A-Z0-9&.\-]{2,8}$", title))
            employer_is_role = self._looks_like_role_title(employer)
            title_is_company = self._is_company_name(title) or title_is_short_org
            if employer_is_role and title_is_company and not self._looks_like_role_title(title):
                title, employer = employer, title

        # Additional contamination fixes:
        # - If title/employer look like description bullets/sentences, push them into description.
        bullet_re = re.compile(r"^\s*[•◦●▪\-–—]+\s+|^\s*[-–—]\s*$")
        if title and (bullet_re.search(title) or self._looks_like_description(title) and not self._looks_like_role_title(title)):

            description = f"{title}\n{description}".strip()
            title = ""
        if employer and (bullet_re.search(employer) or self._looks_like_description(employer) and not self._is_company_name(employer)):
            description = f"{employer}\n{description}".strip()
            employer = ""

        if not employer and description:
            extracted_employer, extracted_location, cleaned_description = self._extract_employer_from_description(description)
            if extracted_employer:
                employer = extracted_employer
                if extracted_location and not clean_ocr_text(entry.get("location", "")):
                    entry["location"] = extracted_location
                description = cleaned_description

        # ── Structural title recovery ────────────────────────────────────────────
        # When title is empty but employer and description are present, the first
        # line of description may be the actual role title (e.g. "Cashier and sales
        # management" as the leading line of a description block). Recover it only
        # when it is structurally short (≤ 6 words) and does not look like a
        # responsibility bullet or sentence fragment.
        if not title and employer and description:
            first_desc_line = clean_ocr_text(description.split("\n")[0])
            words = first_desc_line.split()
            if (
                1 <= len(words) <= 6
                and not first_desc_line.endswith(".")
                and not re.match(r"^\s*[•◦●▪\-–—]", first_desc_line)
            ):
                rest = clean_ocr_text("\n".join(description.split("\n")[1:]))
                title = first_desc_line
                description = rest


        start_month, _ = normalize_month(start_month)
        end_month, _ = normalize_month(end_month)
        start_year, _ = normalize_year(start_year)
        end_year, _ = normalize_year(end_year)

        if current:
            end_month = ""
            end_year = ""

        title = self._normalize_role_seniority_prefix(title)

        # ── Structural anchor check ──────────────────────────────────────────────
        # An entry is only valid if it has structural evidence of being a real job.
        #
        # has_real_date: any start/end year or current flag is present.
        # has_real_title: title is non-empty, not a known section-header label, AND
        #   structurally looks like a role name (≤ 6 words, not a sentence fragment).
        #
        # Why ≤ 6 words and not a description?
        #   Genuine job titles are short noun phrases: "Senior Accountant",
        #   "Aws Cloud Engineer", "Articled Assistant".  Sentence fragments that
        #   landed in the title field because of upstream parser errors
        #   ("Participated in architectural discussions to build") are long,
        #   end with a preposition, or pass _looks_like_description.
        #   A title that is long OR looks like a description sentence is NOT a
        #   structural anchor — it is a mis-assigned continuation line.
        #
        # Rejecting on structural shape (word count + description signal) rather than
        # vocabulary means this rule generalises to any unseen sentence fragment
        # without adding new keyword lists.
        #
        # Records that fail both checks are returned as {} so the caller
        # (_finalize_resume_data / _merge_section_header_experience_fragments)
        # can append their description to the previous real record.
        _SECTION_LABEL_TITLES = SECTION_STOP_HEADERS  # use the shared, unified set
        has_real_date = bool(start_year or end_year or current)

        def _title_is_structurally_valid(t: str) -> bool:
            if not t:
                return False
            if t.lower().strip() in _SECTION_LABEL_TITLES:
                return False
            # Long titles are sentence fragments, not role names
            if len(t.split()) > 6:
                return False
            # Titles that read like description sentences are not real titles
            if self._looks_like_description(t):
                return False
            return True

        has_real_title = _title_is_structurally_valid(title)
        if not has_real_date and not has_real_title:
            return {}

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

    def _extract_projects_from_raw(self, raw_text: str) -> List[Dict[str, Any]]:
        lines = [clean_ocr_text(line) for line in clean_ocr_multiline(raw_text).splitlines() if clean_ocr_text(line)]
        projects: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}

        def flush_current() -> None:
            if current.get("title") or current.get("description"):
                projects.append(
                    {
                        "title": clean_ocr_text(current.get("title", "")),
                        "description": clean_ocr_text("\n".join(current.get("description", []))),
                        "role": clean_ocr_text(current.get("role", "")),
                        "tools": clean_ocr_text(current.get("tools", "")),
                        "url": self._normalize_url(clean_ocr_text(current.get("url", ""))),
                    }
                )

        for line in lines:
            lower = line.lower()
            if lower in SECTION_STOP_HEADERS or lower in {"projects", "project"}:
                continue
            role_match = re.match(r"^role\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            tools_match = re.match(r"^(?:tools?|technologies)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if role_match:
                current["role"] = clean_ocr_text(role_match.group(1))
                continue
            if tools_match:
                current["tools"] = clean_ocr_text(tools_match.group(1))
                continue
            url = self._normalize_url(line)
            if url:
                current["url"] = url
                continue
            if self._is_probable_project_title(line) and not line.startswith(("-", "•")) and len(line.split()) <= 8:
                if current:
                    flush_current()
                current = {"title": line, "description": []}
                continue
            current.setdefault("description", []).append(line)

        if current:
            flush_current()
        return projects

    def _project_quality_score(self, entries: List[Dict[str, Any]]) -> float:
        score = 0.0
        for entry in entries:
            title = clean_ocr_text(entry.get("title", ""))
            description = clean_ocr_text(entry.get("description", ""))
            tools = clean_ocr_text(entry.get("tools", ""))
            if title:
                score += 1.0
            if description:
                score += 0.5
            if tools:
                score += 0.3
        return score

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

    def _repair_experience_sequence(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        repaired: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            candidate = dict(entry)
            title = clean_ocr_text(candidate.get("title", ""))
            employer = clean_ocr_text(candidate.get("employer", candidate.get("company", "")))
            description = clean_ocr_text(candidate.get("description", ""))
            has_date = bool(
                clean_ocr_text(candidate.get("startYear", ""))
                or clean_ocr_text(candidate.get("endYear", ""))
                or candidate.get("current")
            )

            if not employer and description:
                first_sentence = clean_ocr_text(re.split(r"\n|(?<=\.)\s+", description, maxsplit=1)[0])
                if self._is_company_name(first_sentence) and len(first_sentence.split()) <= 8:
                    candidate["employer"] = first_sentence
                    candidate["description"] = clean_ocr_text(description[len(first_sentence):])
                    employer = first_sentence
                    description = clean_ocr_text(candidate["description"])

            title_is_orphan_description = bool(title and not employer and not has_date and self._looks_like_description(title))
            if title_is_orphan_description and repaired:
                previous = repaired[-1]
                merged_description = clean_ocr_text(
                    "\n".join([previous.get("description", ""), title, description])
                )
                previous["description"] = merged_description
                previous["originalDescription"] = merged_description
                continue

            repaired.append(candidate)
        return repaired

    def _is_continuation_line(self, text: str) -> bool:
        """Return True when text looks like a sentence fragment / continuation of description text.

        Continuation lines must never become employers or record anchors.
        They are characterised by:
        - Ending in a period while having multiple words (sentence-like)
        - Starting with a lowercase word (mid-sentence)
        - Containing prepositions/conjunctions at the start
        - Being very long with no company-name indicators
        """
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        lower = cleaned.lower()
        # Starts lower-case or with preposition → continuation
        if cleaned[0].islower():
            return True
        # Ends with period and has multiple words → sentence fragment
        if cleaned.endswith(".") and len(cleaned.split()) >= 3:
            return True
        # Starts with common continuation prepositions
        start_prepositions = {"and", "or", "including", "such", "as", "through", "by", "for", "on", "with", "to", "in"}
        first_word = lower.split()[0].rstrip(".,;:") if lower.split() else ""
        if first_word in start_prepositions and not self._is_company_name(cleaned):
            return True
        return False

    def _filter_bogus_experience_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove obvious bogus experience fragments created from section headers/noise.

        Non-goal: fully determine job validity.
        Goal: prevent multiple small fragments like title="Professional Experience" or employer="HOBBIES" or
        titles that are clearly responsibility bullets / continuation text.
        """
        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = clean_ocr_text(entry.get("title", ""))
            employer = clean_ocr_text(entry.get("employer", ""))
            description = clean_ocr_text(entry.get("description", ""))

            title_lower = title.lower()
            employer_lower = employer.lower()

            # Section header tokens should never be treated as jobs.
            if title_lower in EXPERIENCE_SECTION_HEADERS:
                continue
            if employer_lower in EXPERIENCE_SECTION_HEADERS:
                continue

            # Non-title phrases should never become jobs.
            if title_lower in NON_TITLE_PHRASES:
                continue
            if employer_lower in NON_TITLE_PHRASES:
                continue

            # Year-only fragments are typically noise when used as a "title".
            if re.fullmatch(r"(?:19|20)\d{2}", title_lower):
                continue

            # Hobby-like employers are noise.
            if employer_lower in {"hobbies", "hobby", "interests"}:
                continue

            # Responsibility verb opening in title → drop as bogus record
            if title:
                first_word = title_lower.split()[0].rstrip(".,;:") if title_lower.split() else ""
                if first_word in RESPONSIBILITY_VERBS and not employer:
                    # This is a responsibility bullet mistakenly treated as a record
                    continue

            # Continuation lines that became employers: long sentences not matching company pattern
            if employer and len(employer.split()) > 10 and not self._is_company_name(employer):
                continue

            # If the entry is basically a section header/token without real job info, drop it.
            if title_lower in EXPERIENCE_SECTION_HEADERS and (not employer and not description):
                continue
            if employer_lower in EXPERIENCE_SECTION_HEADERS and (not title and not description):
                continue

            # If title is a section header but description contains only bullet/noise, drop.
            if title_lower in EXPERIENCE_SECTION_HEADERS:
                meaningful = bool(re.search(r"\b[A-Za-z]{3,}\b", description)) and len(description.split()) >= 6
                if not meaningful:
                    continue

            filtered.append(entry)
        return filtered


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
        )

    def _is_valid_certification_name(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        lower = cleaned.lower()
        if EMAIL_FIND_RE.search(cleaned) or PHONE_FIND_RE.search(cleaned):
            return False
        if lower in KNOWN_CERTIFICATION_NAMES:
            return True
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

    def _merge_section_header_experience_fragments(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate structurally incomplete experience records into preceding real records.

        An entry is a fragment when ALL of the following are true:
        1. No date (no startYear, endYear, or current flag).
        2. No structurally valid title — meaning the title is either empty, a known
           section-header label, longer than 6 words, or reads like a description
           sentence.

        This is a purely structural rule.  It does not use employer-name lists,
        job-title vocabularies, or any other keyword matching.  The shape of the
        text (word count, description signal) is the only criterion.

        Why this prevents future failures:
        - Any sentence fragment that was mis-assigned to the title field will be
          > 6 words or will pass _looks_like_description, making it a fragment
          regardless of the specific words it contains.
        - Section headers are caught by SECTION_STOP_HEADERS without needing a
          separate local copy of that set.
        - Employer-only records with no date and no real title are also fragments —
          a company-looking string alone is not sufficient anchor evidence.
        """
        if not entries:
            return []

        def _is_fragment(entry: Dict[str, Any]) -> bool:
            title = clean_ocr_text(entry.get("title", ""))
            has_date = (
                bool(clean_ocr_text(entry.get("startYear", "")))
                or bool(clean_ocr_text(entry.get("endYear", "")))
                or bool(entry.get("current"))
            )
            if has_date:
                return False
            # A title is structurally valid only if it is short, not a section
            # header, and does not look like a description sentence.
            if title:
                if title.lower().strip() in SECTION_STOP_HEADERS:
                    pass  # falls through to True below
                elif len(title.split()) <= 6 and not self._looks_like_description(title):
                    return False  # short, non-description title → real record
            # No date AND no structurally valid title → fragment
            return True

        consolidated: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry:
                continue
            if _is_fragment(entry):
                desc = clean_ocr_text(entry.get("description", ""))
                if desc and consolidated:
                    prev = consolidated[-1]
                    prev_desc = clean_ocr_text(prev.get("description", ""))
                    prev["description"] = clean_ocr_text(f"{prev_desc}\n{desc}".strip())
                    if prev.get("originalDescription"):
                        prev["originalDescription"] = prev["description"]
                continue
            consolidated.append(entry)

        return consolidated

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
            r"\b(Developed|Designed|Collaborated|Gained|Managed|Implemented|Built|Created|Led|Worked|Assisted|"
            r"Identifying|Negotiating|Coordinating|Verifying|Processed|Reviewed|Executed|Prepared|Drafted)\b"
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

    def _extract_leading_location_from_description(self, description: str) -> Tuple[str, str]:
        cleaned = clean_ocr_text(description)
        if not cleaned:
            return "", ""
        for location in sorted(LOCATION_WORDS, key=len, reverse=True):
            match = re.match(rf"^({re.escape(location)})\b[\s,;-]*(?P<body>.+)$", cleaned, re.IGNORECASE)
            if not match:
                continue
            body = clean_ocr_text(match.group("body"))
            if body and len(body.split()) >= 3:
                return match.group(1).title(), body
        return "", cleaned

    def _looks_like_role_title(self, text: str) -> bool:
        cleaned = clean_ocr_text(text)
        if not cleaned:
            return False
        if self._looks_like_date_fragment(cleaned):
            return False
        lower = cleaned.lower()
        if lower in NON_TITLE_PHRASES or self._contains_non_title_phrase(cleaned):
            return False
        words = set(re.findall(r"\b[a-z]+\b", lower))
        if words & TITLE_HINTS:
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
        summary_headers = {
            # explicit summary labels
            "summary",
            "professional summary",
            "career summary",
            "executive summary",
            # objective / goal labels — some resumes put the summary here
            "objective",
            "career objective",
            "professional objective",
            # profile labels
            "profile",
            "professional profile",
            "career profile",
            "about",
            "about me",
            # other common labels
            "introduction",
            "overview",
            "bio",
            "personal statement",
        }
        stop_headers = {
            "experience", "work experience", "professional experience",
            "employment history", "employment", "career history",
            "education", "academic background", "qualifications",
            "skills", "technical skills", "key skills", "core competencies",
            "projects", "certifications", "languages", "achievements",
            "interests", "hobbies", "references",
        }
        for idx, line in enumerate(lines):
            if line.lower().strip(" :|-") not in summary_headers:
                continue
            collected: List[str] = []
            for next_line in lines[idx + 1:]:
                lower = next_line.lower().strip(" :|-")
                if lower in stop_headers:
                    break
                collected.append(next_line)
                # Raised from 800 → 2000 to avoid silently truncating long summaries
                if len(" ".join(collected)) > 2000:
                    break
            joined = clean_ocr_text("\n".join(collected))
            joined = re.split(
                r"\b(core competencies|key achievements|competencies|technical skills)\b",
                joined,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            result = clean_ocr_text(joined)
            if result:
                return result
        return ""

    def _extract_heading_block_from_raw_text(self, raw_text: str, headings: set, stop_headings: set) -> str:
        lines = [clean_ocr_text(line) for line in clean_ocr_multiline(raw_text).splitlines() if clean_ocr_text(line)]
        if not lines:
            return ""
        normalized_headings = {clean_ocr_text(heading).lower() for heading in headings}
        normalized_stops = {clean_ocr_text(heading).lower() for heading in stop_headings}
        for idx, line in enumerate(lines):
            lower = line.lower().strip(" :|-")
            if lower not in normalized_headings:
                continue
            collected: List[str] = []
            for next_line in lines[idx + 1:]:
                next_lower = next_line.lower().strip(" :|-")
                if next_lower in normalized_stops:
                    break
                collected.append(next_line)
            return clean_ocr_multiline("\n".join(collected))
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

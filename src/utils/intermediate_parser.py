import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from src.utils.headings import canonicalize_heading


NOISE_REPLACEMENTS = {
    "\r\n": "\n",
    "\r": "\n",
    "â€¢": "-",
    "•": "-",
    "◦": "-",
    "●": "-",
    "▪": "-",
    "â€“": "-",
    "â€”": "-",
    "–": "-",
    "—": "-",
    "\t": " ",
}

ROLE_KEYWORDS = (
    "engineer",
    "developer",
    "analyst",
    "manager",
    "consultant",
    "intern",
    "lead",
    "architect",
    "specialist",
    "director",
)

DEGREE_KEYWORDS = (
    "bachelor",
    "master",
    "phd",
    "b.tech",
    "m.tech",
    "b.e",
    "m.e",
    "mba",
    "diploma",
)

SECTION_PATTERNS = {
    "skills": [
        r"\b(python|java|javascript|sql|react|node|aws|docker|kubernetes)\b",
        r"\b(framework|library|tool|technology|platform)\b",
    ],
    "education": [
        r"\b(university|college|institute|school)\b",
        r"\b(bachelor|master|phd|degree|gpa|cgpa)\b",
        r"\b(19|20)\d{2}\b",
    ],
    "experience": [
        r"\b(19|20)\d{2}\b",
        r"\b(present|current|responsible|developed|managed|implemented)\b",
        r"\b(company|organization|team|role|project)\b",
    ],
    "projects": [
        r"\b(project|developed|built|implemented|github|api|application)\b",
    ],
    "certifications": [
        r"\b(certificate|certification|certified|aws|azure|gcp|pmp|scrum)\b",
    ],
}


@dataclass
class ParsedSection:
    heading: str = ""
    lines: List[str] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0
    valid: bool = False


@dataclass
class IntermediateParseResult:
    clean_text: str
    sections: Dict[str, ParsedSection]
    unmatched_lines: List[str]


def _normalize_text(text: str) -> str:
    normalized = text or ""
    for bad, good in NOISE_REPLACEMENTS.items():
        normalized = normalized.replace(bad, good)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    return normalized


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[\u2022\-\*\u25CF\u25E6]+\s*", "- ", line)
    line = re.sub(r"^-\s{2,}", "- ", line)
    return line


def _is_probable_heading(line: str) -> Optional[str]:
    if not line:
        return None
    if len(line) > 72:
        return None
    if re.search(r"[@\\]", line):
        return None
    return canonicalize_heading(line)


def _looks_like_heading_candidate(line: str) -> bool:
    if not line:
        return False
    if _is_probable_heading(line):
        return True
    if len(line) > 64:
        return False
    if line.startswith("- "):
        return False
    if re.search(r"[@\\]", line):
        return False
    if line.endswith((".", ";", ",")):
        return False

    words = [w for w in re.split(r"\s+", line.strip()) if w]
    if not words or len(words) > 6:
        return False

    lower = line.lower()
    heading_terms = (
        "summary",
        "objective",
        "profile",
        "skills",
        "competencies",
        "education",
        "experience",
        "employment",
        "projects",
        "certification",
        "achievements",
        "awards",
        "references",
        "languages",
        "interests",
        "declaration",
        "training",
        "technologies",
    )
    if any(term in lower for term in heading_terms):
        return True

    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if not alpha_words:
        return False
    title_like = all(w[0].isupper() for w in alpha_words if w[0].isalpha())
    all_caps = line.upper() == line and any(ch.isalpha() for ch in line)
    return title_like or all_caps


def _is_wrapped_continuation(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if _is_probable_heading(current):
        return False
    if _looks_like_heading_candidate(current):
        return False
    if current.startswith("- "):
        return False
    if re.match(r"^(19|20)\d{2}", current):
        return False
    if re.match(r"^[A-Z][A-Z ]+$", current) and len(current) < 36:
        return False
    if previous.endswith(":"):
        return False
    if previous.endswith("-"):
        return True
    if current[:1].islower():
        return True
    if current.startswith(("and ", "or ", "with ", "to ", "for ", "in ")):
        return True
    return False


def _reconstruct_lines(lines: List[str]) -> List[str]:
    rebuilt: List[str] = []
    for line in lines:
        cleaned = _clean_line(line)
        if not cleaned:
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            continue

        if rebuilt and rebuilt[-1] and _is_wrapped_continuation(rebuilt[-1], cleaned):
            if rebuilt[-1].endswith("-"):
                rebuilt[-1] = rebuilt[-1][:-1] + cleaned
            else:
                rebuilt[-1] = f"{rebuilt[-1]} {cleaned}"
            continue

        rebuilt.append(cleaned)
    return rebuilt


def _is_experience_entry_start(line: str) -> bool:
    lower = line.lower()
    if re.search(r"\b(19|20)\d{2}\b", lower):
        return True
    if any(keyword in lower for keyword in ROLE_KEYWORDS):
        return True
    if re.search(r"\b(at|@)\s+[A-Z]", line):
        return True
    return False


def _is_education_entry_start(line: str) -> bool:
    lower = line.lower()
    if any(keyword in lower for keyword in DEGREE_KEYWORDS):
        return True
    if any(word in lower for word in ("university", "college", "institute", "school")):
        return True
    if re.search(r"\b(19|20)\d{2}\b", lower) and len(line) < 80:
        return True
    return False


def _chunk_entries(section: str, lines: List[str]) -> List[str]:
    if section not in {"experience", "education"}:
        return []
    if not lines:
        return []

    chunks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if not line:
            if current:
                chunks.append(current)
                current = []
            continue

        is_start = (
            _is_experience_entry_start(line)
            if section == "experience"
            else _is_education_entry_start(line)
        )
        if current and is_start:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append(current)

    entries = ["\n".join(block).strip() for block in chunks if block]
    return [entry for entry in entries if len(entry) > 2]


def _semantic_confidence(section: str, lines: List[str], entries: List[str]) -> float:
    content_lines = [line for line in lines if line]
    if not content_lines:
        return 0.0

    text = "\n".join(content_lines)
    word_count = len(text.split())
    structure_score = min(word_count / 80.0, 1.0) * 0.25

    pattern_score = 0.0
    patterns = SECTION_PATTERNS.get(section, [])
    if patterns:
        hits = sum(
            1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE)
        )
        pattern_score = (hits / len(patterns)) * 0.50

    entry_score = 0.0
    if section in {"experience", "education"}:
        entry_score = min(len(entries) / 3.0, 1.0) * 0.25
    elif section in {"skills", "projects", "certifications"}:
        bullets = sum(1 for line in content_lines if line.startswith("- "))
        separators = sum(1 for line in content_lines if "," in line or "|" in line)
        entry_score = min((bullets + separators) / 5.0, 1.0) * 0.25

    return round(min(structure_score + pattern_score + entry_score, 1.0), 2)


def _dedupe_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for line in lines:
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        key = line.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


def _build(text: str) -> IntermediateParseResult:
    normalized_text = _normalize_text(text)
    raw_lines = normalized_text.split("\n")
    reconstructed = _reconstruct_lines(raw_lines)

    sections: Dict[str, ParsedSection] = {}
    unmatched: List[str] = []
    current_section: Optional[str] = None

    for line in reconstructed:
        if not line:
            if current_section:
                sections[current_section].lines.append("")
            elif unmatched and unmatched[-1] != "":
                unmatched.append("")
            continue

        heading = _is_probable_heading(line)
        if heading:
            current_section = heading
            if heading not in sections:
                sections[heading] = ParsedSection(heading=line)
            continue

        if current_section:
            sections[current_section].lines.append(line)
        else:
            unmatched.append(line)

    for section_name, payload in sections.items():
        payload.lines = _dedupe_lines(payload.lines)
        payload.entries = _chunk_entries(section_name, payload.lines)
        if payload.entries:
            payload.text = "\n\n".join(payload.entries).strip()
        else:
            payload.text = "\n".join(line for line in payload.lines if line).strip()
        payload.confidence = _semantic_confidence(
            section_name,
            payload.lines,
            payload.entries,
        )
        payload.valid = bool(payload.text and payload.confidence >= 0.2)

    return IntermediateParseResult(
        clean_text=normalized_text,
        sections=sections,
        unmatched_lines=[line for line in unmatched if line],
    )


@lru_cache(maxsize=48)
def build_intermediate_resume(text: str) -> IntermediateParseResult:
    return _build(text or "")

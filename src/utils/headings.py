import logging
import re
from typing import Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HEADING_ALIASES: Dict[str, List[str]] = {
    "name": [
        "name",
        "personal details",
        "personal information",
    ],
    "summary": [
        "summary",
        "professional summary",
        "career summary",
        "career objective",
        "objective",
        "profile",
        "about me",
        "about",
        "introduction",
    ],
    "skills": [
        "skills",
        "technical skills",
        "key skills",
        "core skills",
        "skill set",
        "skills summary",
        "technical expertise",
        "technical competencies",
        "technical proficiencies",
        "technical knowledge",
        "tools and technologies",
        "tools & technologies",
        "technologies",
        "software skills",
        "programming skills",
        "technology stack",
        "tech stack",
        "core competencies",
        "competencies",
        "professional skills",
        "areas of expertise",
        "computer skills",
        "it skills",
        "soft skills",
    ],
    "education": [
        "education",
        "education qualification",
        "educational background",
        "educational qualifications",
        "academic qualifications",
        "education and qualifications",
        "education & qualifications",
        "education details",
        "education background",
        "academic background",
        "educational profile",
        "educational summary",
        "academic credentials",
        "education and training",
        "qualification",
        "qualifications",
        "degree",
        "degrees",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "employment",
        "work history",
        "job history",
        "career history",
        "internship",
        "internships",
        "professional background",
        "career timeline",
    ],
    "projects": [
        "projects",
        "project",
        "project details",
        "project work",
        "project experience",
        "project details",
        "key projects",
        "academic projects",
        "personal projects",
    ],
    "certifications": [
        "certifications",
        "certification",
        "certificates",
        "professional certifications",
        "technical certifications",
        "certifications and training",
        "certifications & training",
        "training and certifications",
        "training & certifications",
        "licenses and certifications",
        "licenses & certifications",
        "credentials",
        "professional credentials",
        "courses and certifications",
        "certifications and courses",
        "certifications & courses",
        "certifications/courses",
    ],
    "awards": [
        "awards",
        "achievements",
        "select achievements",
        "key achievements",
        "honors",
        "recognition",
        "accomplishments",
        "awards and achievements",
    ],
    "languages": [
        "languages",
        "languages known",
        "language skills",
        "language proficiency",
        "known languages",
    ],
    "interests": [
        "interests",
        "interests & hobbies",
        "hobbies",
        "personal interests",
        "activities",
    ],
    "references": [
        "references",
        "referees",
        "recommendations",
        "professional references",
        "references upon request",
    ],
    "publications": [
        "publications",
        "papers",
        "research",
        "journal articles",
    ],
    "volunteer": [
        "volunteer",
        "volunteering",
        "community service",
        "social work",
    ],
    "training": [
        "training",
        "trainings attended",
        "workshops",
        "seminars",
        "courses",
    ],
    "patents": [
        "patents",
        "inventions",
    ],
    "presentations": [
        "presentations",
        "talks",
        "lectures",
    ],
    "affiliations": [
        "affiliations",
        "memberships",
    ],
    "service": [
        "service",
    ],
    "strengths": [
        "strengths",
        "core strengths",
    ],
    "declaration": [
        "declaration",
        "statement",
    ],
    "contact": [
        "contact",
        "personal profile",
        "contact details",
        "contact information",
    ],
}


def get_section_mapping():
    return {
        "Name": "name",
        "Summary": "summary",
        "Skills": "skills",
        "Technical Skills": "technical skills",
        "Soft Skills": "soft skills",
        "Education": "education",
        "Experience": "experience",
        "Projects": "projects",
        "Certifications": "certifications",
        "Awards": "awards",
        "Languages": "languages",
        "Interests": "interests",
        "References": "references",
        "Publications": "publications",
        "Volunteer": "volunteer",
        "Training": "training",
        "Patents": "patents",
        "Presentations": "presentations",
        "Affiliations": "affiliations",
        "Service": "service",
        "Strengths": "strengths",
        "Declaration": "declaration",
        "Full Resume": "fulltext",
    }


SECTION_HEADINGS = list(get_section_mapping().values())
MAIN_SECTION_HEADINGS = sorted(
    {alias for aliases in HEADING_ALIASES.values() for alias in aliases},
    key=len,
    reverse=True,
)


def _normalize_heading_line(line: str) -> str:
    line = line.strip().lower()
    line = line.replace("â€¢", "-").replace("•", "-")
    line = line.replace("â€“", "-").replace("â€”", "-").replace("–", "-").replace("—", "-")
    line = re.sub(r"^[\-\*\.:;|_ ]+", "", line)
    line = re.sub(r"[\-\*\.:;|_ ]+$", "", line)
    line = re.sub(r"\s+", " ", line)
    return line


def _is_noise_heading_candidate(line: str) -> bool:
    if not line:
        return True
    if len(line) > 64:
        return True
    if re.search(r"[@\\]", line):
        return True
    if re.fullmatch(r"[\d\W_]+", line):
        return True
    return False


ALIAS_TAIL_TOKENS = {
    "details",
    "detail",
    "summary",
    "section",
    "profile",
    "background",
    "history",
    "overview",
    "highlights",
    "highlight",
    "information",
    "and",
    "&",
}


def _has_allowed_alias_tail(tail: str) -> bool:
    if not tail:
        return True
    tokens = [tok for tok in re.split(r"[\s/&]+", tail.strip()) if tok]
    if not tokens or len(tokens) > 2:
        return False
    return all(tok in ALIAS_TAIL_TOKENS for tok in tokens)


def canonicalize_heading(line: str) -> Optional[str]:
    normalized = _normalize_heading_line(line)
    if _is_noise_heading_candidate(normalized):
        return None

    for section, aliases in HEADING_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_heading_line(alias)

            if normalized == alias_norm:
                return section

            if normalized.startswith(alias_norm + " "):
                tail = normalized[len(alias_norm):].strip()
                if _has_allowed_alias_tail(tail):
                    return section

            if re.match(rf"^{re.escape(alias_norm)}\s*[:\-|]\s*$", normalized):
                return section

    return None


def normalize_heading(heading_text: str) -> str:
    canonical = canonicalize_heading(heading_text)
    if canonical:
        return canonical
    fallback = _normalize_heading_line(heading_text).replace(" ", "_")[:24]
    return f"custom_{fallback}" if fallback else "custom_unknown"


def calculate_headings_confidence(
    detected_headings: List[Tuple[str, str]],
    text: str,
) -> float:
    if not detected_headings:
        return 0.0

    total = len(detected_headings)
    unique_sections = len({section for section, _ in detected_headings})
    unique_titles = len({raw.lower().strip() for _, raw in detected_headings})

    diversity_factor = min(unique_sections / 6.0, 1.0) * 0.40
    quality_factor = (unique_titles / max(total, 1)) * 0.35
    length_factor = min(len(text) / 4000.0, 1.0) * 0.25

    confidence = diversity_factor + quality_factor + length_factor
    return round(min(confidence, 1.0), 2)


def _has_following_content(lines: List[str], index: int) -> bool:
    for lookahead in range(index + 1, min(index + 5, len(lines))):
        probe = lines[lookahead].strip()
        if not probe:
            continue
        if canonicalize_heading(probe):
            return False
        if len(probe) > 2:
            return True
    return False


def detect_headings(text: str):
    if not text:
        return [], 0.0

    lines = text.splitlines()
    detected: List[Tuple[str, str]] = []
    seen = set()

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        canonical = canonicalize_heading(line)
        if not canonical:
            continue

        if not _has_following_content(lines, idx):
            continue

        dedupe_key = (canonical, _normalize_heading_line(line))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        detected.append((canonical, line))

    confidence = calculate_headings_confidence(detected, text)
    logger.info("Detected headings: %s, Confidence: %.2f", detected, confidence)
    return detected, confidence


if __name__ == "__main__":
    headings = detect_headings("")
    print("Detected headings:", headings)

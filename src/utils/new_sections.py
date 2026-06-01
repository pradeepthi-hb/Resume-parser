import re
import logging
from src.utils.headings import canonicalize_heading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _apply_corrections(field_name: str, value: str, confidence: float):
    # Runtime correction replay is intentionally disabled in section extractors.
    return value, confidence




LANGUAGE_HEADERS = [
    "languages", "language skills", "known languages", "language proficiency",
    "linguistic skills", "communication skills", "spoken languages"
]

LANGUAGE_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "about", "interests", "hobbies", "references", "declaration"
]


COMMON_LANGUAGES = [
    "english", "hindi", "telugu", "tamil", "kannada", "malayalam", "marathi", "gujarati",
    "bengali", "punjabi", "urdu", "french", "german", "spanish", "portuguese", "italian",
    "chinese", "japanese", "korean", "arabic", "russian", "dutch", "swedish", "norwegian",
    "danish", "finnish", "polish", "turkish", "vietnamese", "thai", "indonesian", "malay",
    "tagalog", "swahili", "hebrew", "greek", "latin", "sanskrit"
]

LANGUAGE_PROFICIENCY_LEVELS = [
    "native", "fluent", "conversational", "professional", "working",
    "intermediate", "beginner", "elementary", "limited", "basic"
]


def calculate_languages_confidence(languages_text: str, original_text: str) -> float:
    if not languages_text or not languages_text.strip():
        return 0.0
    
    lines = [l.strip() for l in languages_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.4
    
    
    valid_count = 0
    for line in lines:
        line_lower = line.lower()
        if any(lang in line_lower for lang in COMMON_LANGUAGES):
            valid_count += 1
    
    valid_factor = min(valid_count / 3, 1.0) * 0.4
    
    
    length_factor = min(len(languages_text) / 100, 1.0) * 0.2
    
    confidence = item_factor + valid_factor + length_factor
    return round(min(confidence, 1.0), 2)


def extract_languages_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    languages_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in LANGUAGE_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            for stop in LANGUAGE_STOP_KEYWORDS:
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            
            if any(lang in line_lower for lang in COMMON_LANGUAGES):
                languages_lines.append(line_stripped)
    
    
    cleaned = []
    seen = set()
    for line in languages_lines:
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    languages_text = "\n".join(cleaned)
    confidence = calculate_languages_confidence(languages_text, text)
    
    logger.info(f"Languages extraction completed. Found: {len(cleaned)}, Confidence: {confidence}")
    
    languages_text, confidence = _apply_corrections("languages", languages_text, confidence)
    return languages_text, confidence







INTERESTS_HEADERS = [
    "interests", "hobbies", "personal interests", "activities", "interest",
    "hobby", "leisure activities", "pastimes"
]

INTERESTS_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "references", "declaration", "languages",
    "personal profile", "languages known", "email", "mobile", "contact"
]


def calculate_interests_confidence(interests_text: str, original_text: str) -> float:
    if not interests_text or not interests_text.strip():
        return 0.0
    
    lines = [l.strip() for l in interests_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.5
    
    
    length_factor = min(len(interests_text) / 100, 1.0) * 0.5
    
    confidence = item_factor + length_factor
    return round(min(confidence, 1.0), 2)


def extract_interests_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    interests_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in INTERESTS_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            for stop in INTERESTS_STOP_KEYWORDS:
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            interests_lines.append(line_stripped)
    
    
    cleaned = []
    seen = set()
    for line in interests_lines:
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    interests_text = "\n".join(cleaned)
    confidence = calculate_interests_confidence(interests_text, text)
    
    logger.info(f"Interests extraction completed. Found: {len(cleaned)}, Confidence: {confidence}")
    return interests_text, confidence




ACHIEVEMENTS_HEADERS = [
    "achievements", "awards", "honors", "recognition", "accomplishments",
    "awards & achievements", "awards and achievements", "honors & awards"
]

ACHIEVEMENTS_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "references", "declaration", "languages", "interests",
    "professional experience", "work experience", "employment history",
    "tools & technologies", "tools and technologies", "additional information"
]


def calculate_achievements_confidence(achievements_text: str, original_text: str) -> float:
    if not achievements_text or not achievements_text.strip():
        return 0.0
    
    lines = [l.strip() for l in achievements_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.4
    
    
    achievement_patterns = [
        r'\b(awarded|won|received|achieved|earned|granted)\b',
        r'\b(prize|medal|certificate|recognition|trophy)\b',
        r'\b(winner|first|second|third|top|best)\b'
    ]
    
    pattern_matches = 0
    for pattern in achievement_patterns:
        pattern_matches += len(re.findall(pattern, achievements_text, re.IGNORECASE))
    
    pattern_factor = min(pattern_matches / 3, 1.0) * 0.3
    
    
    length_factor = min(len(achievements_text) / 150, 1.0) * 0.3
    
    confidence = item_factor + pattern_factor + length_factor
    return round(min(confidence, 1.0), 2)


def extract_achievements_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    achievements_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in ACHIEVEMENTS_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            for stop in ACHIEVEMENTS_STOP_KEYWORDS:
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            achievements_lines.append(line_stripped)
    
    
    cleaned = []
    seen = set()
    for line in achievements_lines:
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    achievements_text = "\n".join(cleaned)
    confidence = calculate_achievements_confidence(achievements_text, text)
    
    logger.info(f"Achievements extraction completed. Found: {len(cleaned)}, Confidence: {confidence}")
    return achievements_text, confidence




PUBLICATIONS_HEADERS = [
    "publications", "papers", "research", "conference papers", "journal articles",
    "thesis", "dissertation", "books", "presentations"
]

PUBLICATIONS_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "references", "declaration", "awards"
]


def calculate_publications_confidence(publications_text: str, original_text: str) -> float:
    if not publications_text or not publications_text.strip():
        return 0.0
    
    lines = [l.strip() for l in publications_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.4
    
    
    pub_patterns = [
        r'\b(published|presented|conference|journal|paper|article)\b',
        r'\b(research|study|analysis)\b',
        r'\b(isbn|doi|volume|issue)\b'
    ]
    
    pattern_matches = 0
    for pattern in pub_patterns:
        pattern_matches += len(re.findall(pattern, publications_text, re.IGNORECASE))
    
    pattern_factor = min(pattern_matches / 3, 1.0) * 0.3
    
    
    length_factor = min(len(publications_text) / 200, 1.0) * 0.3
    
    confidence = item_factor + pattern_factor + length_factor
    return round(min(confidence, 1.0), 2)


def extract_publications_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    publications_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in PUBLICATIONS_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            for stop in PUBLICATIONS_STOP_KEYWORDS:
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            publications_lines.append(line_stripped)
    
    
    cleaned = []
    seen = set()
    for line in publications_lines:
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    publications_text = "\n".join(cleaned)
    confidence = calculate_publications_confidence(publications_text, text)
    
    logger.info(f"Publications extraction completed. Found: {len(cleaned)}, Confidence: {confidence}")
    return publications_text, confidence




VOLUNTEER_HEADERS = [
    "volunteer", "volunteering", "community service", "social work",
    "charity", "community involvement", "social activities"
]

VOLUNTEER_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "references", "declaration", "awards"
]


def calculate_volunteer_confidence(volunteer_text: str, original_text: str) -> float:
    if not volunteer_text or not volunteer_text.strip():
        return 0.0
    
    lines = [l.strip() for l in volunteer_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.4
    
    
    volunteer_patterns = [
        r'\b(volunteer|community|charity|non-profit|social)\b',
        r'\b(organization|foundation|trust|society)\b'
    ]
    
    pattern_matches = 0
    for pattern in volunteer_patterns:
        pattern_matches += len(re.findall(pattern, volunteer_text, re.IGNORECASE))
    
    pattern_factor = min(pattern_matches / 2, 1.0) * 0.3
    
    
    length_factor = min(len(volunteer_text) / 150, 1.0) * 0.3
    
    confidence = item_factor + pattern_factor + length_factor
    return round(min(confidence, 1.0), 2)


def extract_volunteer_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    volunteer_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in VOLUNTEER_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            for stop in VOLUNTEER_STOP_KEYWORDS:
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            volunteer_lines.append(line_stripped)
    
    
    cleaned = []
    seen = set()
    for line in volunteer_lines:
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    volunteer_text = "\n".join(cleaned)
    confidence = calculate_volunteer_confidence(volunteer_text, text)
    
    logger.info(f"Volunteer extraction completed. Found: {len(cleaned)}, Confidence: {confidence}")
    return volunteer_text, confidence




SUMMARY_HEADERS = [
    "summary", "career summary", "professional summary", "objective",
    "career objective", "profile", "about me", "about", "introduction"
]

SUMMARY_STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "references", "declaration"
]

# Extended patterns to detect skills section headers in summary
SKILLS_SECTION_PATTERNS = [
    r'^(?:technical\s+)?skills\s*[:\-–—]?\s*$',
    r'^(?:programming\s+languages?|technologies?|tools?|soft\s+skills)\s*[:\-–—]?\s*$',
    r'^(?:technical\s+)?skills\s*$',
    r'^(?:programming|web\s+development|database|tools)\s*$',
]


def _is_skills_section_header(line: str) -> bool:
    """Check if a line is a skills section header."""
    line_lower = line.lower().strip()
    canonical = canonicalize_heading(line)
    if canonical and canonical != "summary":
        return True
    
    # Check exact matches
    if line_lower in SUMMARY_STOP_KEYWORDS:
        return True
    
    # Check heading-like startswith (avoid stopping on prose sentences)
    for stop in SUMMARY_STOP_KEYWORDS:
        if re.match(rf'^{re.escape(stop)}\s*[:\-â€“â€”]?\s*$', line_lower):
            return True
    
    # Check regex patterns
    for pattern in SKILLS_SECTION_PATTERNS:
        if re.match(pattern, line_lower, re.IGNORECASE):
            return True
    
    # Check if line contains "skills" as a header (not in the middle of text)
    if re.search(r'^(?:technical\s+)?skills\s*[:\-–—]?\s*$', line_lower, re.IGNORECASE):
        return True
    
    # Check for skill category headers like "Programming Languages:", "Web Development:", etc.
    skill_categories = ['programming languages', 'web development', 'database', 'tools', 'soft skills']
    for cat in skill_categories:
        if line_lower.startswith(cat):
            return True
    
    return False


def calculate_summary_confidence(summary_text: str, original_text: str) -> float:
    if not summary_text or not summary_text.strip():
        return 0.0
    
    
    length = len(summary_text)
    if 50 <= length <= 500:
        length_factor = 0.6
    elif length < 50:
        length_factor = 0.3
    else:
        length_factor = max(0.5, 1.0 - (length - 500) / 1000)
    
    
    professional_patterns = [
        r'\b(experienced|skilled|professional|dedicated|motivated)\b',
        r'\b(years|experience|expertise|proficient)\b'
    ]
    
    pattern_matches = 0
    for pattern in professional_patterns:
        pattern_matches += len(re.findall(pattern, summary_text, re.IGNORECASE))
    
    pattern_factor = min(pattern_matches / 3, 1.0) * 0.4
    
    confidence = length_factor + pattern_factor
    return round(min(confidence, 1.0), 2)


def extract_summary_from_resume(text: str) -> tuple:
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    summary_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        
        is_header = False
        for header in SUMMARY_HEADERS:
            if line_lower == header:
                is_header = True
                break
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
        
        if is_header:
            capture = True
            header_found = True
            continue
        
        
        if capture and header_found:
            should_stop = False
            heading_key = canonicalize_heading(line_stripped)
            if heading_key and heading_key != "summary":
                should_stop = True
            
            # Also check for skills section headers using the extended pattern detection
            if not should_stop and _is_skills_section_header(line_stripped):
                should_stop = True
            
            if should_stop:
                break
        
        if capture and header_found and line_stripped:
            summary_lines.append(line_stripped)
    
    
    if not summary_lines:
        fallback_lines = []
        for line in lines[:60]:
            line_stripped = line.strip()
            if not line_stripped:
                if fallback_lines:
                    break
                continue
            if canonicalize_heading(line_stripped):
                if fallback_lines:
                    break
                continue
            line_lower = line_stripped.lower()
            if "@" in line_lower or re.search(r'\+?\d[\d\s\-\(\)]{8,}', line_lower):
                continue
            if len(line_stripped.split()) < 4:
                continue
            fallback_lines.append(line_stripped)
            if len(" ".join(fallback_lines)) >= 320:
                break
        summary_lines = fallback_lines

    summary_text = " ".join(summary_lines)
    confidence = calculate_summary_confidence(summary_text, text)
    
    
    summary_text, confidence = _apply_corrections("summary", summary_text, confidence)
    
    logger.info(f"Summary extraction completed. Confidence: {confidence}")
    return summary_text, confidence


if __name__ == "__main__":
    
    sample_text = ""
    print("Languages:", extract_languages_from_resume(sample_text))
    print("Interests:", extract_interests_from_resume(sample_text))
    print("Achievements:", extract_achievements_from_resume(sample_text))
    print("Publications:", extract_publications_from_resume(sample_text))
    print("Volunteer:", extract_volunteer_from_resume(sample_text))
    print("Summary:", extract_summary_from_resume(sample_text))

import re
import logging
from pdfminer.high_level import extract_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SKILL_HEADERS = [
    "skills", "technical skills", "key skills", "core skills", "skill set",
    "skills summary", "technical expertise", "technical competencies", 
    "technical proficiencies", "technical knowledge", "tools & technologies", 
    "tools and technologies", "technologies", "software skills", 
    "programming skills", "technology stack", "tech stack",
    "core competencies", "competencies", "professional skills", 
    "areas of expertise", "strengths", "skills & abilities", 
    "skills and abilities", "computer skills", "it skills",
    "technical skill", "skill set", "skills set","soft skills",
    
    "professional competencies", "technical abilities", "skill summary",
    "expertise", "technical profile", "skill profile", "technical skills & competencies"
]


STOP_KEYWORDS = [
    "education", "projects", "experience", "certifications",
    "summary", "about", "technical skills", "strength", "strengths",
    "personal details", "extra-curricular activities", "languages",
    "interests", "hobbies", "references", "declaration", "objective",
    "career objective", "introduction", "contact", "achievements",
    "awards", "publications", "volunteer experience", "internships",
    "work experience", "employment", "professional experience",
    "awards & achievements", "training", "trainings attended",
    "educational background", "education background", "project details",
    "personal profile", "languages known", "internship"
]

ROLE_ORG_PATTERN = re.compile(
    r"\b(assistant|accountant|engineer|developer|manager|analyst|executive|specialist|intern|co\.|ltd|llp|inc)\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b|(?:\d{1,2}/\d{4}\s*[-–]\s*\d{1,2}/\d{4})", re.IGNORECASE)
SENTENCE_VERB_PATTERN = re.compile(
    r"\b(reviewed|managed|assisted|supported|developed|implemented|coordinated|authored|prepared|drafted|collaborated|achieved|reducing|improving)\b",
    re.IGNORECASE,
)
SKILLS_HEADING_NOISE = {"key achievements", "achievements", "core competencies", "languages", "education", "experience"}
SKILLS_AWARD_NOISE_PATTERN = re.compile(r"\b(success|boost|excellence|award|winner|achiev)\b", re.IGNORECASE)
ALLOWED_SINGLE_WORD_SKILLS = {
    "budgeting",
    "forecasting",
    "regulatory",
    "compliance",
    "taxation",
    "auditing",
    "accounting",
    "analysis",
    "excel",
    "python",
    "java",
    "sql",
}


def get_skills_keywords():
    return SKILL_HEADERS


def get_skills_stop_keywords():
    return STOP_KEYWORDS


def calculate_skills_confidence(skills_text, original_text):
    if not skills_text or not skills_text.strip():
        return 0.0
    
    lines = [l.strip() for l in skills_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 5, 1.0) * 0.5  
    
    
    length_factor = min(len(skills_text) / 200, 1.0) * 0.5
    
    confidence = item_factor + length_factor
    
    
    if confidence > 0.3 and len(skills_text) > 20:
        confidence = min(confidence * 1.2, 1.0)
    
    return round(min(confidence, 1.0), 2)


def _normalize_skill_line(line: str) -> str:
    line = line.strip()
    line = line.replace("\ue01c", "").strip()
    line = re.sub(r"\s{2,}", " ", line)
    return line


def _looks_like_skill_line(line: str) -> bool:
    if not line or len(line) < 2:
        return False

    line = _normalize_skill_line(line)
    lower = line.lower()
    words = line.split()
    lowered_clean = re.sub(r"[^a-z ]", "", lower).strip()

    if lowered_clean in SKILLS_HEADING_NOISE:
        return False

    if DATE_PATTERN.search(line):
        return False

    if SKILLS_AWARD_NOISE_PATTERN.search(lower):
        return False

    if ROLE_ORG_PATTERN.search(lower):
        return False

    if SENTENCE_VERB_PATTERN.search(lower):
        return False

    if line.startswith("- ") and len(words) > 8:
        return False

    # Long prose sentences are rarely skills.
    if len(words) > 10 and ":" not in line:
        return False

    if "," in line and ":" not in line and len(words) > 3:
        return False

    if "." in line and ":" not in line:
        return False

    if line.endswith((".", ";")):
        return False

    if len(words) == 1:
        token = re.sub(r"[^A-Za-z]", "", words[0])
        token_lower = token.lower()
        if token_lower not in ALLOWED_SINGLE_WORD_SKILLS and re.match(r"^[A-Z][a-z]+$", token):
            return False

    # Keep category lines (e.g., "Languages: Java, C")
    if ":" in line:
        return True

    # Keep short competency phrases.
    if len(words) <= 7:
        return True

    return False


def extract_skills_from_resume(text):
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    skills_lines = []
    capture = False
    header_found = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()

        
        is_header = False
        for header in SKILL_HEADERS:
            
            if line_lower == header:
                is_header = True
                break
            
            if len(line_stripped) < 50:
                if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
                
                if header in line_lower and len(line_stripped) < 40:
                    header_pos = line_lower.find(header)
                    if header_pos == 0 or (header_pos > 0 and line_stripped[header_pos-1] in ' -:'):
                        is_header = True
                        break

        if is_header:
            capture = True
            header_found = True
            continue

        
        if capture and header_found:
            should_stop = False
            for stop in STOP_KEYWORDS:
                
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
                
                if line_lower == stop:
                    should_stop = True
                    break

                if stop in line_lower and len(line_lower) < 40:
                    should_stop = True
                    break
            
            if should_stop:
                capture = False
                header_found = False
                continue
        
        
        if capture and header_found and line_stripped and not re.fullmatch(r"-{4,}", line_stripped):
            skills_lines.append(line_stripped)

    
    cleaned = []
    seen = set()
    separator_pattern = re.compile(r'^[-–—=_*#]+$')
    
    for line in skills_lines:
        
        if separator_pattern.match(line):
            continue
        
        normalized = _normalize_skill_line(line)
        if normalized.startswith("& ") and cleaned:
            cleaned[-1] = f"{cleaned[-1]} {normalized}".strip()
            seen.add(cleaned[-1].lower())
            continue

        if not _looks_like_skill_line(normalized):
            continue

        key = normalized.lower()
        if key not in seen:
            cleaned.append(normalized)
            seen.add(key)

    skills_text = "\n".join(cleaned)
    
    
    confidence = calculate_skills_confidence(skills_text, text)
    
    logger.info(f"Skills extraction completed. Items found: {len(cleaned)}, Confidence: {confidence}")
    
    return skills_text, confidence



if __name__ == "__main__":
    pdf_path = "sample_resume.pdf"
    text = extract_text(pdf_path)

    skills_section, confidence = extract_skills_from_resume(text)

    print("=== Skills Section ===")
    print(skills_section if skills_section else "No skills section found")
    print(f"Confidence: {confidence}")

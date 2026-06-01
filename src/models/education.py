import re
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EDUCATION_HEADERS = [
    "education", "education qualification", "educational qualifications",
    "academic qualifications", "education & qualifications", "education details",
    "education background", "educational background", "academic background", "educational profile",
    "educational summary", "academic credentials", "qualification", "qualifications",
    "academic details", "education details", "degree", "degrees"
]


STOP_KEYWORDS = [
    "skills", "projects", "certifications", "experience", "summary",
    "about", "technical skills", "work experience", "employment",
    "internships", "projects", "achievements", "awards",
    "objective", "career objective", "project details", "training", "trainings attended"
]


def calculate_education_confidence(education_text, original_text):
    if not education_text or not education_text.strip():
        return 0.0
    
    lines = [l.strip() for l in education_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 5, 1.0) * 0.4  
    
    
    edu_patterns = [
        r'\b(bachelor|master|phd|doctorate|diploma|certificate)\b',
        r'\b(20\d{2}|19\d{2})\b',  
        r'\b(university|college|institute|school)\b',
        r'\b(gpa|grade|percentage|score)\b',
    ]
    
    edu_matches = 0
    for pattern in edu_patterns:
        edu_matches += len(re.findall(pattern, education_text, re.IGNORECASE))
    
    edu_factor = min(edu_matches / 8, 1.0) * 0.4  
    
    
    length_factor = min(len(education_text) / 300, 1.0) * 0.2
    
    confidence = item_factor + edu_factor + length_factor
    return round(confidence, 2)


def extract_education_from_resume(text):
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    
    education_lines = []
    capture = False
    header_found = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        lower = line_stripped.lower()

        
        is_header = False
        for header in EDUCATION_HEADERS:
            
            if lower == header:
                is_header = True
                break
            
            if len(line_stripped) < 50:
                if lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', lower):
                    is_header = True
                    break
                
                if header in lower and len(line_stripped) < 40:
                    header_pos = lower.find(header)
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
                
                if lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', lower):
                    should_stop = True
                    break
                
                if lower == stop:
                    should_stop = True
                    break
            
            if should_stop:
                break
            
            education_lines.append(line_stripped)

    
    cleaned = []
    seen = set()
    separator_pattern = re.compile(r'^[-–—=_*#]+$')
    year_pattern = re.compile(r'^(19|20)\d{2}(\s*[-–]\s*(19|20)\d{2})?$')
    
    for line in education_lines:
        
        if len(line) < 3:
            continue
        
        if separator_pattern.match(line):
            continue
        
        if year_pattern.fullmatch(line):
            continue
        
        if line not in seen:
            cleaned.append(line)
            seen.add(line)

    education_text = "\n".join(cleaned)
    
    
    confidence = calculate_education_confidence(education_text, text)
    
    logger.info(f"Education extraction completed. Entries found: {len(cleaned)}, Confidence: {confidence}")
    
    return education_text, confidence

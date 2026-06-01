import re
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TRAINING_HEADERS = [
    "training", "trainings", "workshops", "seminars", "courses", 
    "professional development", "training & development", "training programs",
    "workshops & seminars", "training courses", "corporate training",
    "skill development", "learning & development", "trainings attended",
    "internship", "internships"
]


STOP_KEYWORDS = [
    "skills", "education", "experience", "projects", "certifications",
    "summary", "about", "interests", "hobbies", "references", "declaration",
    "languages", "awards", "achievements", "publications"
]


TRAINING_KEYWORDS = [
    r'\b(training|workshop|seminar|course|certified|completed)\b',
    r'\b(attended|participated|undergone|undertook)\b',
    r'\b(certificate|diploma|program|bootcamp)\b'
]


def calculate_training_confidence(training_text, original_text):
    if not training_text or not training_text.strip():
        return 0.0
    
    lines = [l.strip() for l in training_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    
    
    item_count = len(lines)
    item_factor = min(item_count / 3, 1.0) * 0.4
    
    
    training_matches = 0
    for pattern in TRAINING_KEYWORDS:
        training_matches += len(re.findall(pattern, training_text, re.IGNORECASE))
    
    pattern_factor = min(training_matches / 3, 1.0) * 0.4
    
    
    length_factor = min(len(training_text) / 150, 1.0) * 0.2
    
    confidence = item_factor + pattern_factor + length_factor
    
    if confidence > 0.3 and len(training_text) > 20:
        confidence = min(confidence * 1.2, 1.0)
    
    return round(min(confidence, 1.0), 2)


def is_training_header(line_stripped, line_lower):
    for header in TRAINING_HEADERS:
        
        if line_lower == header:
            return True
        
        
        if len(line_stripped) < 50:
            if line_lower.startswith(header) or re.match(rf'^{re.escape(header)}[\s:\-–—]+', line_lower):
                return True
            
            if header in line_lower and len(line_stripped) < 40:
                header_pos = line_lower.find(header)
                if header_pos == 0 or (header_pos > 0 and line_stripped[header_pos-1] in ' -:'):
                    return True
    
    return False


def should_stop_extraction(line_lower, capture_count):
    for stop in STOP_KEYWORDS:
        
        if line_lower == stop or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
            return True
        
        
        if stop in line_lower and len(line_lower) < 30 and capture_count > 0:
            return True
    
    return False


def extract_training_from_resume(text):
    if not text:
        return "", 0.0
    
    lines = text.splitlines()
    
    training_lines = []
    capture = False
    header_found = False
    capture_count = 0

    
    separator_pattern = re.compile(r'^[-–—=_*#]+$')
    year_pattern = re.compile(r'^(19|20)\d{2}(\s*[-–]\s*(19|20)\d{2})?$')
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()

        
        is_header = is_training_header(line_stripped, line_lower)

        if is_header:
            capture = True
            header_found = True
            continue

        
        if capture and header_found:
            
            if should_stop_extraction(line_lower, capture_count):
                break
            
            
            if separator_pattern.match(line_stripped):
                continue
            
            
            if year_pattern.fullmatch(line_stripped):
                continue
            
            
            if len(line_stripped) < 3:
                continue
            
            training_lines.append(line_stripped)
            capture_count += 1

    
    cleaned = []
    seen = set()
    
    for line in training_lines:
        normalized = line.lower().strip()
        
        
        if normalized in seen:
            continue
        if separator_pattern.match(line):
            continue
            
        cleaned.append(line)
        seen.add(normalized)

    training_text = "\n".join(cleaned)
    
    confidence = calculate_training_confidence(training_text, text)
    
    logger.info(f"Training extraction completed. Entries found: {len(cleaned)}, Confidence: {confidence}")
    
    return training_text, confidence


if __name__ == "__main__":
    sample_text = ""
    print("Training:", extract_training_from_resume(sample_text))

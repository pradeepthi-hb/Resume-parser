import pdfminer
import re
import os
import logging

from pdfminer.high_level import extract_text
from src.utils.name_database import (
    INDIAN_FIRST_NAMES,
    INDIAN_SURNAMES,
    WESTERN_FIRST_NAMES,
    GLOBAL_SURNAMES,
    INVALID_NAME_STRINGS,
    NAME_PARTICLES,
    NON_NAME_PATTERNS
)


NAME_TITLE_PREFIXES = [
    r'^(?:Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Master|Sir|Madam|Lord|Lady)\s+',
    r'^(?:Er\.?|Er\s+)',
    r'^(?:C\.?A\.?|CA)\s+',
    r'^(?:M\.?Tech\.?|B\.?Tech\.?|Ph\.?D\.?|M\.?Sc\.?|B\.?Sc\.?)\s+',
]

NAME_SUFFIXES = [
    r'(?:Jr\.?|Sr\.?|II|III|IV|V)$',
    r'(?:\s+(?:Jr|Sr|II|III|IV|V))$',
]

COMPOUND_NAME_PATTERNS = [
    r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)+$',
    r'^[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+[A-Z][a-z]+$',
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+[A-Z][a-z]+$",
]


try:
    import fitz  
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None


try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    NLP_AVAILABLE = True
except ImportError:
    nlp = None
    NLP_AVAILABLE = False


try:
    from layoutlm_extractor import extract_with_layoutlm, is_layoutlm_available
    LAYOUTLM_AVAILABLE = is_layoutlm_available()
except ImportError:
    LAYOUTLM_AVAILABLE = False
    extract_with_layoutlm = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"


TOP_SECTION_Y_THRESHOLD = 300






def is_known_first_name(name_part):
    if not name_part:
        return False
    
    name_lower = name_part.lower().strip()
    
    
    if name_lower in INDIAN_FIRST_NAMES:
        return True
    
    
    if name_lower in WESTERN_FIRST_NAMES:
        return True
    
    
    for known_name in INDIAN_FIRST_NAMES:
        if name_lower == known_name.lower():
            return True
    
    for known_name in WESTERN_FIRST_NAMES:
        if name_lower == known_name.lower():
            return True
    
    return False


def is_known_surname(name_part):
    if not name_part:
        return False
    
    name_lower = name_part.lower().strip()
    
    
    if name_lower in INDIAN_SURNAMES:
        return True
    
    
    if name_lower in GLOBAL_SURNAMES:
        return True
    
    return False


def has_name_awareness(name):
    if not name or not name.strip():
        return False, 0.0
    
    name_parts = name.strip().split()
    
    if len(name_parts) < 1:
        return False, 0.0
    
    
    name_lower = name.lower()
    for invalid in INVALID_NAME_STRINGS:
        if invalid in name_lower:
            logger.info(f"Name '{name}' rejected - contains invalid string: {invalid}")
            return False, 0.0
    
    
    score = 0.0
    max_score = 1.0
    
    
    known_first_count = 0
    known_surname_count = 0
    
    for part in name_parts:
        if is_known_first_name(part):
            known_first_count += 1
        if is_known_surname(part):
            known_surname_count += 1
    
    
    total_parts = len(name_parts)
    
    if total_parts == 1:
        
        if known_first_count > 0:
            score = 0.7
        else:
            
            if re.match(r'^[A-Z][a-z]{1,15}$', name):
                score = 0.4
            else:
                score = 0.1
    else:
        
        
        if known_first_count > 0 and known_surname_count > 0:
            score = 0.9
        
        elif known_first_count > 0 or known_surname_count > 0:
            score = 0.7
        else:
            
            
            valid_structure = True
            for part in name_parts:
                if not re.match(r'^[A-Z][a-z]{1,15}$', part):
                    valid_structure = False
                    break
            
            if valid_structure and 2 <= total_parts <= 4:
                score = 0.5
            else:
                score = 0.2
    
    is_valid = score >= 0.4
    return is_valid, round(score, 2)


def validate_name_with_awareness(name, text):
    if not name:
        return False, 0.0
    
    
    if not is_valid_name(name, text):
        
        is_aware, awareness_score = has_name_awareness(name)
        if is_aware and awareness_score > 0.5:
            logger.info(f"Name '{name}' passed via name awareness (score: {awareness_score})")
            return True, awareness_score
        return False, awareness_score
    
    
    is_aware, awareness_score = has_name_awareness(name)
    
    
    if is_aware:
        return True, min(awareness_score + 0.2, 1.0)
    
    
    return True, 0.6






def extract_text_with_coordinates(pdf_path):
    if not PYMUPDF_AVAILABLE or fitz is None:
        logger.warning("PyMuPDF not available, falling back to pdfminer")
        return None, None
    
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning(f"PDF path not found: {pdf_path}")
        return None, None
    
    try:
        logger.info(f"Extracting text with coordinates from: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        full_text = ""
        text_blocks = []  
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            full_text += text + "\n"
            
            
            blocks = page.get_text("blocks")
            for block in blocks:
                
                if len(block) >= 5:
                    text_block = {
                        'text': block[4].strip(),
                        'x0': block[0],
                        'y0': block[1],
                        'x1': block[2],
                        'y1': block[3],
                        'page': page_num
                    }
                    if text_block['text']:
                        text_blocks.append(text_block)
        
        doc.close()
        logger.info(f"Extracted {len(text_blocks)} text blocks from {len(doc)} pages")
        return full_text, text_blocks
        
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        return None, None






def filter_top_section(text_blocks, y_threshold=TOP_SECTION_Y_THRESHOLD):
    if not text_blocks:
        return ""
    
    top_blocks = []
    for block in text_blocks:
        
        if block.get('page', 0) == 0:
            
            if block['y1'] <= y_threshold:
                top_blocks.append(block)
    
    
    top_blocks.sort(key=lambda b: b['y0'])
    
    
    top_text = "\n".join([b['text'] for b in top_blocks])
    logger.info(f"Top section extracted: {len(top_blocks)} blocks, {len(top_text)} chars")
    return top_text


def extract_name_from_top_section(pdf_path, text):
    if not PYMUPDF_AVAILABLE:
        return None, 0.0, -1
    
    
    full_text, text_blocks = extract_text_with_coordinates(pdf_path)
    
    if not text_blocks:
        logger.info("No text blocks found with PyMuPDF")
        return None, 0.0, -1
    
    
    top_section_text = filter_top_section(text_blocks, TOP_SECTION_Y_THRESHOLD)
    
    if not top_section_text or len(top_section_text.strip()) < 10:
        logger.info("Top section too short, using full text")
        top_section_text = text[:1500] if text else ""
    
    
    if NLP_AVAILABLE and nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except:
            pass
    
    if NLP_AVAILABLE and nlp is not None:
        try:
            doc = nlp(top_section_text[:1500])
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    ent_text = ent.text.strip()
                    
                    
                    if re.search(r'\d{4}', ent_text):
                        continue
                    if re.search(r'(to|-)', ent_text, re.IGNORECASE):
                        continue
                    
                    words = ent_text.split()
                    if 2 <= len(words) <= 4:
                        capital_count = sum(1 for w in words if w and w[0].isupper())
                        if capital_count >= len(words) * 0.7:
                            candidate_name = ent_text.title()
                            
                            
                            is_valid, awareness_score = validate_name_with_awareness(candidate_name, top_section_text)
                            if is_valid:
                                position = top_section_text.find(ent_text)
                                line_number = top_section_text[:position].count('\n') if position >= 0 else 0
                                confidence = calculate_name_confidence(candidate_name, top_section_text, "nlp", line_number)
                                
                                
                                confidence = min(confidence + awareness_score * 0.2, 1.0)
                                
                                if confidence >= 0.5:
                                    logger.info(f"PyMuPDF+NER extracted name: {candidate_name}, Confidence: {confidence}")
                                    return candidate_name, confidence, line_number
            
            logger.info("No valid name found in top section using NER")
            
        except Exception as e:
            logger.warning(f"NER on top section failed: {e}")
    
    
    name, confidence, position = extract_name_using_rules(top_section_text)
    if name and confidence > 0:
        logger.info(f"PyMuPDF+Rules extracted name: {name}, Confidence: {confidence}")
        return name, confidence, position
    
    
    name, confidence, position = extract_name_using_heuristics(top_section_text)
    if name and confidence > 0:
        logger.info(f"PyMuPDF+Heuristic extracted name: {name}, Confidence: {confidence}")
        return name, confidence, position
    
    logger.info("No name found in top section, falling back to full text")
    return None, 0.0, -1


SECTION_HEADERS_TO_FILTER = [
    "career objective", "objective", "summary", "professional summary", "career summary",
    "about me", "about", "introduction", "profile",
    "key skills", "skills", "technical skills", "skill set", "core skills",
    "education", "education qualification", "academic qualifications", "educational qualifications",
    "experience", "work experience", "employment history", "professional experience",
    "projects", "project experience", "academic projects", "personal projects",
    "certifications", "certificates", "credentials", "licenses",
    "awards", "achievements", "honors", "recognition",
    "languages", "language skills", "interests", "hobbies",
    "references", "declaration", "contact", "contact details",
    "personal details", "personal information", "site engineer",
    "software developer", "software engineer", "data scientist", "data analyst",
    "web developer", "full stack developer", "front end developer", "back end developer",
    "project manager", "team lead", "technical lead", "manager", "consultant",
    "analyst", "designer", "architect", "administrator", "coordinator",
    "intern", "trainee", "junior", "senior", "lead", "head",
    "dynamic site supervisor", "supervisor", "foreman",
    "java script", "web development", "python developer", "react developer",
    "nagarjuna sagar", "uppuguda hyderabad", "anantapur dist", "hyderabad",
    "of firm", "firm name", "company name"
]


JOB_TITLES_TO_FILTER = [
    "site engineer", "software developer", "software engineer", "data scientist",
    "data analyst", "web developer", "full stack developer", "fullstack developer",
    "front end developer", "front-end developer", "back end developer", "back-end developer",
    "project manager", "team lead", "technical lead", "manager", "consultant",
    "business analyst", "system analyst", "qa engineer", "test engineer",
    "devops engineer", "cloud engineer", "network engineer", "security engineer",
    "ui designer", "ux designer", "graphic designer", "designer",
    "solution architect", "system architect", "enterprise architect",
    "database administrator", "dba", "system administrator", "sysadmin",
    "recruiter", "hr manager", "hr executive", "coordinator",
    "intern", "trainee", "junior developer", "senior developer",
    "lead developer", "tech lead", "head of", "director", "vp",
    "chief", "executive", "president", "ceo", "cto", "cfo",
    "supervisor", "foreman", "superintendent",
    "assistant", "associate", "officer", "clerk",
    "operator", "technician", "specialist", "expert", "professional"
]


LOCATION_PATTERNS = [
    r'\b(hyderabad|chennai|bangalore|mumbai|delhi|kolkata|pune|ahmedabad)\b',
    r'\b(anantapur|visakhapatnam|vijayawada|guntur|warangal|karimnagar)\b',
    r'\b(uppuguda|secunderabad|kukatpally|miyapur|ameerpet)\b',
    r'\b(telangana|andhra pradesh|karnataka|tamil nadu|maharashtra)\b',
    r'\b(india|usa|uk|canada|australia|germany|france|japan|china)\b',
    r'\b(mandal|district|state|city|town|village)\b',
    r'\b(nagarjuna sagar)\b'
]


DATE_PATTERNS = [
    r'^\d{4}\s*(to|-)\s*\d{4}$',  
    r'^\w+\s+\d{4}\s+(to|-)\s+(\w+\s+)?\d{4}$',  
    r'^\d{1,2}/\d{1,2}/\d{4}$',  
    r'^\w+\s+\d{4}$',  
    r'^\d{4}$',  
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s+(to|-)\s+(january|february|march|april|may|june|july|august|september|october|november|december)?\s*\d{4}\b',  
    r'\b\d{4}\s*-\s*\d{4}\b',  
]


INVALID_NAME_PATTERNS = [
    r'\b(plant|operations|production|maintenance|quality|engineering)\b',
    r'\b(motivated|dedicated|skilled|experienced|professional)\b',
    r'\b(engineer|developer|manager|analyst|designer|consultant)\b',
    r'\b(engineering|development|management|analysis|design)\b',
]


INDIAN_NAME_PATTERNS = [
    r'\b(kumar|krishna|rao|reddy|naidu|prasad|babu|ji|sinh)\b',
    r'\b(pvt|ltd|inc|corp|co)\b',
]


VALID_NAME_COMPONENTS = [
    r'\b[A-Z][a-z]{1,15}\b',  
    r'\b[A-Z]{2,}\b',  
]


def is_valid_name(name, text):
    if not name:
        return False
    
    name_lower = name.lower().strip()
    
    
    for pattern in DATE_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            logger.info(f"Name '{name}' filtered - matches date pattern: {pattern}")
            return False
    
    
    for header in SECTION_HEADERS_TO_FILTER:
        if header in name_lower:
            logger.info(f"Name '{name}' filtered - contains section header: {header}")
            return False
    
    
    for title in JOB_TITLES_TO_FILTER:
        if title in name_lower:
            logger.info(f"Name '{name}' filtered - is job title: {title}")
            return False
    
    
    for pattern in LOCATION_PATTERNS:
        if re.search(pattern, name_lower):
            logger.info(f"Name '{name}' filtered - matches location pattern: {pattern}")
            return False
    
    
    lines = text.splitlines()[:10]
    name_found_early = False
    for i, line in enumerate(lines):
        if name.lower() in line.lower():
            name_found_early = True
            
            if i < 5:
                return True
            break
    
    if name_found_early:
        logger.info(f"Name '{name}' filtered - not in first 5 lines")
        return False
    
    return True


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def calculate_name_confidence(name, text, method_used="regex", position=0):
    if not name or not name.strip():
        return 0.0
    
    confidence = 0.0
    
    
    method_scores = {
        "layoutlm": 0.30,
        "pymupdf_ner": 0.28,
        "nlp": 0.25,
        "regex": 0.20,
        "heuristic": 0.15
    }
    confidence += method_scores.get(method_used, 0.15)
    
    
    
    if position < 3:
        confidence += 0.25
    elif position < 5:
        confidence += 0.20
    elif position < 8:
        confidence += 0.15
    else:
        confidence += 0.10
    
    
    name_parts = name.split()
    if 2 <= len(name_parts) <= 4:
        confidence += 0.15
        
        valid_parts = 0
        for part in name_parts:
            if re.match(r'^[A-Z][a-z]{1,15}$', part):
                valid_parts += 1
            elif re.match(r'^[A-Z]{2,}$', part):
                valid_parts += 1
        if valid_parts == len(name_parts):
            confidence += 0.10
    
    
    lines = text.splitlines()[:10]
    for i, line in enumerate(lines):
        if name.lower() in line.lower():
            if i < 3:
                confidence += 0.20
            elif i < 6:
                confidence += 0.15
            else:
                confidence += 0.10
            break
    
    return round(min(confidence, 1.0), 2)



def extract_name_using_layoutlm(pdf_path):
    if not LAYOUTLM_AVAILABLE or extract_with_layoutlm is None:
        return None, 0.0
    
    if not pdf_path or not os.path.exists(pdf_path):
        return None, 0.0
    
    try:
        logger.info("Attempting name extraction using LayoutLM")
        result, layoutlm_confidence = extract_with_layoutlm(pdf_path)
        
        if result and "error" not in result:
            name = result.get("name")
            if name and is_valid_name(name, ""):
                
                adjusted_confidence = min(layoutlm_confidence * 1.2, 1.0) if layoutlm_confidence > 0 else 0.75
                logger.info(f"LayoutLM extracted name: {name}, Confidence: {adjusted_confidence}")
                return name, round(adjusted_confidence, 2)
        
        logger.info("LayoutLM did not find name or extraction failed")
        return None, 0.0
        
    except Exception as e:
        logger.warning(f"LayoutLM name extraction failed: {e}")
        return None, 0.0




def extract_name_using_nlp(text):
    if not NLP_AVAILABLE or nlp is None:
        return None, 0.0, -1
    
    try:
        
        doc = nlp(text[:1500])
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent_text = ent.text.strip()
                
                
                if re.search(r'\d{4}', ent_text):  
                    continue
                if re.search(r'(to|-)', ent_text, re.IGNORECASE):  
                    continue
                
                words = ent_text.split()
                
                if 2 <= len(words) <= 4:
                    
                    capital_count = sum(1 for w in words if w and w[0].isupper())
                    if capital_count >= len(words) * 0.7:  
                        candidate_name = ent_text.title()
                        
                        
                        position = text.find(ent_text)
                        line_number = text[:position].count('\n') if position >= 0 else -1
                        
                        
                        is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
                        if is_valid:
                            confidence = calculate_name_confidence(candidate_name, text, "nlp", line_number)
                            confidence = min(confidence + awareness_score * 0.2, 1.0)
                            if confidence >= 0.5:
                                logger.info(f"NLP extracted name: {candidate_name}, Confidence: {confidence}")
                                return candidate_name, confidence, line_number
        
        logger.info("NLP did not find valid person name")
        return None, 0.0, -1
        
    except Exception as e:
        logger.warning(f"NLP name extraction failed: {e}")
        return None, 0.0, -1




def extract_name_using_rules(text):
    lines = text.splitlines()[:15]  
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        
        
        if re.search(r'[\w\.-]+@[\w\.-]+', line):  
            continue
        if re.search(r'\d{10,}', line):  
            continue
        if re.search(r'\d{4}\s*(to|-)\s*\d{4}', line, re.IGNORECASE):  
            continue
        
        
        if re.match(r'^(resume|cv|curriculum\s+vitae|profile|summary|objective|contact)$', line, re.IGNORECASE):
            continue
        
        
        name_with_initial = re.search(r'^([A-Z]\.?\s*)?([A-Z]{2,})\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?$', line)
        if name_with_initial:
            parts = []
            if name_with_initial.group(1):
                parts.append(name_with_initial.group(1).replace('.', '').strip())
            if name_with_initial.group(2):
                parts.append(name_with_initial.group(2).title())
            if name_with_initial.group(3):
                parts.append(name_with_initial.group(3).title())
            if name_with_initial.group(4):
                parts.append(name_with_initial.group(4).title())
            
            if len(parts) >= 2:
                candidate_name = ' '.join(parts)
                
                is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
                if is_valid:
                    confidence = calculate_name_confidence(candidate_name, text, "regex", i)
                    confidence = min(confidence + awareness_score * 0.2, 1.0)
                    logger.info(f"Rule-based (initial) extracted name: {candidate_name}, Confidence: {confidence}")
                    return candidate_name, confidence, i
        
        
        all_caps_name = re.search(r'^([A-Z]{2,})\s+([A-Z]{2,})(?:\s+([A-Z]{2,}))?$', line)
        if all_caps_name:
            parts = []
            if all_caps_name.group(1):
                parts.append(all_caps_name.group(1).title())
            if all_caps_name.group(2):
                parts.append(all_caps_name.group(2).title())
            if all_caps_name.group(3):
                parts.append(all_caps_name.group(3).title())
            
            if len(parts) >= 2:
                candidate_name = ' '.join(parts)
                is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
                if is_valid:
                    confidence = calculate_name_confidence(candidate_name, text, "regex", i)
                    confidence = min(confidence + awareness_score * 0.2, 1.0)
                    logger.info(f"Rule-based (all-caps) extracted name: {candidate_name}, Confidence: {confidence}")
                    return candidate_name, confidence, i
        
        
        standard_name = re.search(r'^([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})(?:\s+([A-Z][a-z]{1,15}))?$', line)
        if standard_name:
            candidate_name = standard_name.group(0)
            is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
            if is_valid:
                confidence = calculate_name_confidence(candidate_name, text, "regex", i)
                confidence = min(confidence + awareness_score * 0.2, 1.0)
                logger.info(f"Rule-based (standard) extracted name: {candidate_name}, Confidence: {confidence}")
                return candidate_name, confidence, i
        
        
        name_match = re.search(r'(?:name|candidate|applicant)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line, re.IGNORECASE)
        if name_match:
            candidate_name = name_match.group(1).title()
            is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
            if is_valid:
                confidence = calculate_name_confidence(candidate_name, text, "regex", i)
                confidence = min(confidence + awareness_score * 0.2, 1.0)
                logger.info(f"Rule-based (label) extracted name: {candidate_name}, Confidence: {confidence}")
                return candidate_name, confidence, i
    
    return None, 0.0, -1






def extract_name_using_heuristics(text):
    lines = text.splitlines()[:20]
    
    
    indian_patterns = [
        r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+(?:Kumar|Reddy|Naidu|Prasad|Rao|Krishna|Babu)$',
        r'^(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'^([A-Z][a-z]+)\s+([A-Z])\.\s+([A-Z][a-z]+)',
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 3 or len(line) > 50:
            continue
        
        
        if re.search(r'[\w\.-]+@[\w\.-]+|\d{10,}|^\d{4}', line):
            continue
        
        
        for pattern in indian_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                candidate_name = match.group(0)
                is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
                if is_valid:
                    confidence = calculate_name_confidence(candidate_name, text, "heuristic", i)
                    confidence = min(confidence + awareness_score * 0.2, 1.0)
                    logger.info(f"Heuristic extracted name: {candidate_name}, Confidence: {confidence}")
                    return candidate_name, confidence, i
        
        
        words = line.split()
        if 2 <= len(words) <= 3:
            
            if all(w and len(w) > 1 and w[0].isupper() for w in words):
                
                lower_words = [w.lower() for w in words]
                if not any(kw in lower_words for kw in SECTION_HEADERS_TO_FILTER[:10]):
                    candidate_name = ' '.join(words)
                    is_valid, awareness_score = validate_name_with_awareness(candidate_name, text)
                    if is_valid:
                        confidence = calculate_name_confidence(candidate_name, text, "heuristic", i)
                        confidence = min(confidence + awareness_score * 0.2, 1.0)
                        logger.info(f"Heuristic (first line) extracted name: {candidate_name}, Confidence: {confidence}")
                        return candidate_name, confidence, i
    
    return None, 0.0, -1






def extract_name_from_resume(text, pdf_path=None):
    if not text:
        return "", 0.0
    
    
    candidates = []
    
    
    
    if pdf_path and os.path.exists(pdf_path):
        name, confidence = extract_name_using_layoutlm(pdf_path)
        if name and confidence > 0:
            
            is_valid, awareness_score = validate_name_with_awareness(name, text)
            if is_valid:
                confidence = min(confidence + awareness_score * 0.2, 1.0)
                candidates.append((name, confidence, "layoutlm"))
                logger.info(f"LayoutLM candidate: {name} (confidence: {confidence})")
    
    
    
    if pdf_path and os.path.exists(pdf_path):
        name, confidence, position = extract_name_from_top_section(pdf_path, text)
        if name and confidence > 0:
            candidates.append((name, confidence, "pymupdf_ner"))
            logger.info(f"PyMuPDF+NER candidate: {name} (confidence: {confidence})")
    
    
    
    name, confidence, position = extract_name_using_nlp(text)
    if name and confidence > 0:
        candidates.append((name, confidence, "nlp"))
        logger.info(f"NLP candidate: {name} (confidence: {confidence})")
    
    
    
    name, confidence, position = extract_name_using_rules(text)
    if name and confidence > 0:
        candidates.append((name, confidence, "regex"))
        logger.info(f"Regex candidate: {name} (confidence: {confidence})")
    
    
    
    name, confidence, position = extract_name_using_heuristics(text)
    if name and confidence > 0:
        candidates.append((name, confidence, "heuristic"))
        logger.info(f"Heuristic candidate: {name} (confidence: {confidence})")
    
    
    if not candidates:
        logger.warning("No name candidates found using any method")
        return "", 0.0
    
    
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_name, best_confidence, best_method = candidates[0]
    
    
    
    is_valid, awareness_score = validate_name_with_awareness(best_name, text)
    if is_valid:
        final_confidence = min(best_confidence + awareness_score * 0.15, 1.0)
        logger.info(f"Final name extracted: {best_name}, Confidence: {final_confidence}, Method: {best_method}")
        return best_name, round(final_confidence, 2)
    else:
        
        for name, confidence, method in candidates[1:]:
            is_valid, awareness_score = validate_name_with_awareness(name, text)
            if is_valid:
                final_confidence = min(confidence + awareness_score * 0.15, 1.0)
                logger.info(f"Final name extracted (fallback): {name}, Confidence: {final_confidence}, Method: {method}")
                return name, round(final_confidence, 2)
        
        
        logger.warning("All name candidates failed validation")
        return "", 0.0






def extract_name_from_resume_simple(text):
    return extract_name_from_resume(text, None)


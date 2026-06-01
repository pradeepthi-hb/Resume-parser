import re
import os
import logging
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

from src.utils.headings import SECTION_HEADINGS, canonicalize_heading
from src.utils.intermediate_parser import build_intermediate_resume


from src.models.education import extract_education_from_resume
from src.models.skills import extract_skills_from_resume
from src.models.certifications import extract_certifications_from_resume
from src.models.projects import extract_projects_section
from src.models.experience import extract_experience_from_resume
from src.models.awards import extract_awards_from_resume
from src.models.references import extract_references_from_resume
from src.models.declaration import extract_declaration_from_resume
from src.models.contact import extract_contact_from_resume
from src.models.strengths import extract_strengths_from_resume
from src.models.training import extract_training_from_resume
from src.models.extracurricular import extract_extracurricular_from_resume
from src.models.email import extract_email_from_resume, extract_all_emails_from_text
from src.models.phone import extract_phone_from_resume, extract_all_phones_from_text
from src.models.links import extract_links_from_resume, extract_all_links_from_text



def _apply_corrections(field_name: str, value: str, confidence: float) -> Tuple[str, float, Dict[str, Any]]:
    # Runtime correction replay is intentionally disabled at the extractor level.
    # Any safe deterministic correction is applied only at the final pipeline layer.
    return value, confidence, {"applied": False, "reason": "disabled"}


try:
    from src.extractors.layoutlm_extractor import extract_with_layoutlm, is_layoutlm_available
    LAYOUTLM_AVAILABLE = is_layoutlm_available()
except ImportError:
    LAYOUTLM_AVAILABLE = False
    extract_with_layoutlm = None

try:
    from src.extractors.pdf_layout_improved import extract_full_resume_html
    PDF_LAYOUT_CUES_AVAILABLE = True
except ImportError:
    extract_full_resume_html = None
    PDF_LAYOUT_CUES_AVAILABLE = False


try:
    from src.utils.ats_extractor import extract_all_ats_sections, is_likely_ats_format
    ATS_EXTRACTOR_AVAILABLE = True
except ImportError:
    ATS_EXTRACTOR_AVAILABLE = False
    extract_all_ats_sections = None
    is_likely_ats_format = None


LAYOUTLM_CONFIDENCE_THRESHOLD = 0.7  
ATS_CONFIDENCE_THRESHOLD = 0.7  


LAYOUTLM_SECTION_MAP = {
    "name": "name",
    "email": "email",
    "phone": "phone",
    "summary": "summary",
    "skills": "skills",
    "education": "education",
    "experience": "experience",
    "projects": "projects",
    "certifications": "certifications",
    "languages": "languages"
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SECTION_KEYWORDS = {
    "name": ["name", "personal details", "personal information", "contact info", "profile"],
    "summary": ["summary", "career summary", "professional summary", "objective", 
                 "career objective", "profile", "about me", "about", "introduction"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "skill set",
               "skills summary", "technical expertise", "technical competencies", 
               "technical proficiencies", "technologies", "tools & technologies",
               "tools and technologies", "software skills", "programming skills",
               "technology stack", "tech stack", "competencies", "professional skills",
               "areas of expertise", "computer skills", "it skills", "skills & abilities",
               "skills and abilities"],
    "education": ["education", "education qualification", "educational qualifications",
                  "academic qualifications", "education & qualifications", "education details",
                  "education background", "academic background", "educational profile",
                  "educational summary", "academic credentials", "qualification", "qualifications",
                  "academic details", "degree", "degrees"],
    "experience": ["experience", "work experience", "employment history", "employment",
                   "professional experience", "work history", "job history", "career history",
                   "internship", "internships", "practical experience"],
    "projects": ["projects", "project", "academic projects", "personal projects",
                 "project experience", "project details", "key projects", "project work",
                 "major projects", "minor projects"],
    "certifications": ["certifications", "certificates", "certification", "credentials",
                       "licenses", "licenses & certifications", "certifications & licenses",
                       "professional certifications", "certifications earned", 
                       "certifications obtained", "certificate", "certified", "license"],
    "awards": ["awards", "achievements", "honors", "recognition", "accomplishments",
               "awards & achievements", "awards and achievements"],
    "languages": ["languages", "language skills", "language proficiency", "known languages"],
    "interests": ["interests", "hobbies", "personal interests", "activities"],
    "references": ["references", "referees", "recommendations"],
    "declaration": ["declaration", "statement", "legal"],
    "contact": ["contact", "contact details", "contact information", "email", "phone", "address"],
    "publications": ["publications", "papers", "research", "conference papers"],
    "volunteer": ["volunteer", "volunteering", "community service", "social work"],
    "training": ["training", "workshops", "seminars", "courses", "professional development"],
    "strengths": ["strengths", "strength", "key strengths", "personal strengths", "core strengths"],
    "extra-curricular": ["extra-curricular", "extracurricular", "co-curricular"]
}


STOP_KEYWORDS = {
    "skills": ["education", "projects", "experience", "certifications", "summary", 
               "about", "technical skills", "strength", "strengths", "personal details",
               "extra-curricular activities", "languages", "interests", "hobbies", 
               "references", "declaration", "objective", "career objective", 
               "introduction", "contact", "achievements", "awards", "publications",
               "volunteer experience", "internships", "work experience", "employment",
               "professional experience", "awards & achievements"],
    "education": ["skills", "projects", "experience", "certifications", "summary",
                  "about", "technical skills", "work experience", "employment",
                  "internships", "achievements", "awards"],
    "projects": ["education", "experience", "skills", "certifications", "summary",
                 "about", "contact", "references", "achievements", "awards",
                 "languages", "declaration"],
    "experience": ["skills", "projects", "education", "certifications", "summary",
                   "about", "contact", "references", "achievements", "awards",
                   "languages", "interests", "hobbies", "declaration"],
    "certifications": ["skills", "projects", "education", "experience", "summary",
                       "about", "technical skills", "work experience", "employment",
                       "interships", "achievements", "awards", "languages"],
    "summary": ["skills", "projects", "education", "experience", "certifications",
                "about", "technical skills", "work experience", "employment",
                "contact", "references"],
    "awards": ["skills", "projects", "education", "experience", "certifications",
               "summary", "languages", "interests", "references", "declaration"],
    "languages": ["skills", "projects", "education", "experience", "certifications",
                  "summary", "interests", "hobbies", "references", "declaration"],
    "interests": ["skills", "projects", "education", "experience", "certifications",
                  "summary", "references", "declaration"],
    "references": ["declaration"],
    "declaration": [],
    "contact": ["skills", "projects", "education", "experience", "certifications",
               "summary", "references"],
    "publications": ["skills", "projects", "education", "experience", "certifications",
                     "summary", "references"],
    "volunteer": ["skills", "projects", "education", "experience", "certifications",
                  "summary", "references", "declaration"],
    "training": ["skills", "projects", "education", "experience", "certifications",
                 "summary", "references"],
    "strengths": ["skills", "projects", "education", "experience", "certifications",
                 "summary", "references"],
    "name": ["summary", "skills", "education", "experience", "projects",
             "certifications", "contact"]
}


SECTION_PATTERNS = {
    "skills": [
        r'\b(python|java|javascript|html|css|sql|react|angular|vue|node|django|flask)\b',
        r'\b(machine learning|deep learning|data science|artificial intelligence)\b',
        r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git)\b',
        r'\b(c\+\+|c#|ruby|php|swift|kotlin|go|rust)\b',
        r'\b(mysql|postgresql|mongodb|redis|elasticsearch)\b',
        r'\b(framework|library|tool|platform|language)\b'
    ],
    "education": [
        r'\b(bachelor|master|phd|doctorate|diploma|certificate)\b',
        r'\b(20\d{2}|19\d{2})\b',
        r'\b(university|college|institute|school)\b',
        r'\b(gpa|grade|percentage|score)\b',
        r'\b(degree|graduation|passed)\b'
    ],
    "projects": [
        r'\b(developed|built|created|designed|implemented|led)\b',
        r'\b(python|java|javascript|react|angular|node|django|flask)\b',
        r'\b(machine learning|data science|api|database|web)\b',
        r'\b(github|gitlab|heroku|aws|azure)\b',
        r'\b(project|team|application|system)\b'
    ],
    "experience": [
        r'\b(company|organization|employer|work)\b',
        r'\b(20\d{2}|19\d{2})\b',
        r'\b(developer|engineer|manager|analyst|designer|consultant)\b',
        r'\b(intern|junior|senior|lead|head)\b',
        r'\b(responsible|duties|achievements)\b'
    ],
    "certifications": [
        r'\b(certified|certificate|certification)\b',
        r'\b(aws|azure|gcp|google|amazon|microsoft)\b',
        r'\b(pmp|scrum|agile|pmi|itil|ccna|ccnp|mcse|mcsa)\b',
        r'\b(iso|ceh|cissp|cisa|comptia)\b',
        r'\b(license|earned|obtained)\b'
    ],
    "awards": [
        r'\b(awarded|won|received|achieved)\b',
        r'\b(prize|medal|certificate|recognition)\b',
        r'\b(competition|contest|event)\b',
        r'\b(first|second|third|winner)\b'
    ]
}


def _get_isolated_section(text: str, section_type: str) -> Tuple[str, float, bool]:
    try:
        parsed = build_intermediate_resume(text)
        section = parsed.sections.get(section_type.lower())
        if section and section.text:
            return section.text, section.confidence, section.valid
    except Exception as e:
        logger.debug(f"Intermediate parsing failed for '{section_type}': {e}")
    return "", 0.0, False


def _compose_section_scoped_text(section_type: str, isolated_text: str) -> str:
    if not isolated_text:
        return ""
    heading = section_type.replace("-", " ").title()
    return f"{heading}\n{isolated_text}"


def _choose_best_result(
    isolated_text: str,
    isolated_confidence: float,
    extracted_text: str,
    extracted_confidence: float,
) -> Tuple[str, float]:
    if isolated_text and isolated_confidence >= 0.2:
        if not extracted_text:
            return isolated_text, max(isolated_confidence, 0.45)
        if isolated_confidence >= extracted_confidence - 0.15:
            return isolated_text, max(isolated_confidence, extracted_confidence * 0.9)
    return extracted_text, extracted_confidence


def _allow_ats_fallback(
    isolated_text: str,
    isolated_confidence: float,
    current_confidence: float,
) -> bool:
    if isolated_text and isolated_confidence >= 0.25:
        return False
    return current_confidence < ATS_CONFIDENCE_THRESHOLD


def _normalize_layout_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _estimate_layout_split(text_elements) -> Tuple[bool, float]:
    if not text_elements:
        return False, 0.0

    x_mins = [float(getattr(el, "x0", 0.0)) for el in text_elements]
    x_maxs = [float(getattr(el, "x1", 0.0)) for el in text_elements]
    if not x_mins or not x_maxs:
        return False, 0.0

    page_min = min(x_mins)
    page_max = max(x_maxs)
    split_x = page_min + ((page_max - page_min) / 2.0)
    if page_max - page_min < 180:
        return False, split_x

    left = [
        el for el in text_elements
        if float(getattr(el, "x0", 0.0)) <= split_x - 20 and len(_normalize_layout_line(getattr(el, "text", ""))) >= 3
    ]
    right = [
        el for el in text_elements
        if float(getattr(el, "x0", 0.0)) >= split_x + 20 and len(_normalize_layout_line(getattr(el, "text", ""))) >= 3
    ]
    if len(left) < 8 or len(right) < 8:
        return False, split_x

    ratio = min(len(left), len(right)) / max(len(left), len(right))
    return ratio >= 0.25, split_x


def _ordered_layout_elements(text_elements):
    if not text_elements:
        return []

    is_two_column, split_x = _estimate_layout_split(text_elements)
    if not is_two_column:
        return sorted(
            text_elements,
            key=lambda el: (getattr(el, "page_num", 0), -getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)),
        )

    pages = {}
    for el in text_elements:
        page_num = getattr(el, "page_num", 0)
        pages.setdefault(page_num, []).append(el)

    ordered = []
    for page_num in sorted(pages.keys()):
        page_items = pages[page_num]
        left_col = [el for el in page_items if float(getattr(el, "x0", 0.0)) <= split_x]
        right_col = [el for el in page_items if float(getattr(el, "x0", 0.0)) > split_x]
        left_col = sorted(left_col, key=lambda el: (-getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)))
        right_col = sorted(right_col, key=lambda el: (-getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)))

        if not left_col or not right_col:
            ordered.extend(left_col)
            ordered.extend(right_col)
            continue

        page_min_x = min(float(getattr(el, "x0", 0.0)) for el in page_items)
        page_max_x = max(float(getattr(el, "x1", 0.0)) for el in page_items)
        span = max(1.0, page_max_x - page_min_x)
        left_width = max(1.0, max(float(getattr(el, "x1", 0.0)) for el in left_col) - min(float(getattr(el, "x0", 0.0)) for el in left_col))
        right_width = max(1.0, max(float(getattr(el, "x1", 0.0)) for el in right_col) - min(float(getattr(el, "x0", 0.0)) for el in right_col))
        left_text = sum(len(_normalize_layout_line(getattr(el, "text", ""))) for el in left_col)
        right_text = sum(len(_normalize_layout_line(getattr(el, "text", ""))) for el in right_col)

        header_band = []
        left_body = []
        right_body = []
        header_threshold = max(float(getattr(el, "y1", 0.0)) for el in page_items) - 24.0
        for el in page_items:
            if float(getattr(el, "x0", 0.0)) <= split_x:
                target = left_body
            else:
                target = right_body
            if float(getattr(el, "y1", 0.0)) >= header_threshold and float(getattr(el, "x0", 0.0)) <= split_x <= float(getattr(el, "x1", 0.0)):
                header_band.append(el)
            else:
                target.append(el)

        left_body = sorted(left_body, key=lambda el: (-getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)))
        right_body = sorted(right_body, key=lambda el: (-getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)))
        header_band = sorted(header_band, key=lambda el: (-getattr(el, "y1", 0.0), getattr(el, "x0", 0.0)))

        left_ratio = left_width / span
        right_ratio = right_width / span
        left_is_sidebar = left_ratio < 0.38 and left_text < right_text * 0.7
        right_is_sidebar = right_ratio < 0.38 and right_text < left_text * 0.7

        ordered.extend(header_band)
        if left_is_sidebar and not right_is_sidebar:
            ordered.extend(right_body)
            ordered.extend(left_body)
        elif right_is_sidebar and not left_is_sidebar:
            ordered.extend(left_body)
            ordered.extend(right_body)
        else:
            ordered.extend(left_body)
            ordered.extend(right_body)

    logger.info("Two-column PDF layout detected; applying column-aware reading order")
    return ordered


def _build_layout_section_map(text_elements) -> Dict[str, Tuple[str, float]]:
    if not text_elements:
        return {}

    ordered = _ordered_layout_elements(text_elements)
    is_two_column, split_x = _estimate_layout_split(ordered)

    sections: Dict[str, list] = {}
    current_section = None
    current_section_column = None
    section_column_owner: Dict[str, str] = {}

    for el in ordered:
        raw_text = _normalize_layout_line(getattr(el, "text", ""))
        if not raw_text:
            continue

        element_column = "left"
        if is_two_column and float(getattr(el, "x0", 0.0)) > split_x:
            element_column = "right"

        heading = canonicalize_heading(raw_text)
        if heading:
            current_section = heading
            current_section_column = element_column
            section_column_owner.setdefault(heading, element_column)
            sections.setdefault(current_section, [])
            continue

        allowed_column = section_column_owner.get(current_section, current_section_column)
        if current_section and (not is_two_column or allowed_column == element_column):
            sections.setdefault(current_section, []).append(raw_text)

    resolved: Dict[str, Tuple[str, float]] = {}
    for section_name, lines in sections.items():
        clean_lines = []
        seen = set()
        for line in lines:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            clean_lines.append(line)

        text = "\n".join(clean_lines).strip()
        if not text:
            continue

        line_count = len(clean_lines)
        word_count = len(text.split())
        confidence = min(0.35 + min(line_count / 10.0, 0.3) + min(word_count / 180.0, 0.2), 0.85)
        resolved[section_name] = (text, round(confidence, 2))

    return resolved


@lru_cache(maxsize=24)
def _get_layout_section_map(pdf_path: str) -> Dict[str, Tuple[str, float]]:
    if (
        not PDF_LAYOUT_CUES_AVAILABLE
        or not pdf_path
        or not os.path.exists(pdf_path)
        or not str(pdf_path).lower().endswith(".pdf")
    ):
        return {}

    try:
        layout_result = extract_full_resume_html(pdf_path)
        return _build_layout_section_map(layout_result.text_elements)
    except Exception as e:
        logger.debug(f"Layout cue extraction failed for '{pdf_path}': {e}")
        return {}


def _get_layout_cued_section(pdf_path: Optional[str], section_type: str) -> Tuple[str, float]:
    if not pdf_path:
        return "", 0.0
    section_map = _get_layout_section_map(pdf_path)
    return section_map.get(section_type.lower(), ("", 0.0))


def _prefer_layout_cues(
    extracted_text: str,
    extracted_confidence: float,
    layout_text: str,
    layout_confidence: float,
    isolated_text: str = "",
    isolated_confidence: float = 0.0,
) -> Tuple[str, float]:
    if not layout_text:
        return extracted_text, extracted_confidence

    if not extracted_text:
        return layout_text, max(layout_confidence, 0.35)

    if isolated_text and isolated_confidence >= 0.3 and extracted_confidence >= layout_confidence:
        return extracted_text, extracted_confidence

    if layout_confidence >= extracted_confidence + 0.12:
        return layout_text, layout_confidence

    if extracted_confidence < 0.45 and layout_confidence >= extracted_confidence - 0.05:
        if len(layout_text.split()) >= max(3, int(len(extracted_text.split()) * 0.7)):
            return layout_text, max(layout_confidence, extracted_confidence)

    return extracted_text, extracted_confidence


STRICT_HEADING_REQUIRED_SECTIONS = {
    "projects",
    "certifications",
    "awards",
    "references",
    "declaration",
    "strengths",
    "training",
    "extra-curricular",
}


def calculate_section_confidence(section_text, original_text, section_type):
    if not section_text or not section_text.strip():
        return 0.0
    
    lines = [l.strip() for l in section_text.splitlines() if l.strip()]
    
    if not lines:
        return 0.0
    

    item_count = len(lines)
    item_factor = min(item_count / 5, 1.0) * 0.3  
    

    patterns = SECTION_PATTERNS.get(section_type, [])
    pattern_matches = 0
    for pattern in patterns:
        pattern_matches += len(re.findall(pattern, section_text, re.IGNORECASE))
    

    if section_type in ["skills", "education", "experience", "projects", "certifications"]:
        pattern_factor = min(pattern_matches / 5, 1.0) * 0.4
    else:
        pattern_factor = min(pattern_matches / 3, 1.0) * 0.3
    

    length_factor = min(len(section_text) / 200, 1.0) * 0.3
    

    relevance_factor = 0.0
    if section_type in SECTION_KEYWORDS:
        keywords = SECTION_KEYWORDS[section_type]
       
        keyword_count = sum(1 for kw in keywords if kw.lower() in section_text.lower())
        relevance_factor = min(keyword_count / 3, 1.0) * 0.1
    
    confidence = item_factor + pattern_factor + length_factor + relevance_factor

    if confidence > 0.7:
        confidence = min(confidence * 1.1, 1.0)  
    
    return round(min(confidence, 1.0), 2)


def extract_section_by_type(text, section_type):
    if not text:
        return "", 0.0
    
    keywords = SECTION_KEYWORDS.get(section_type.lower(), [])
    stops = STOP_KEYWORDS.get(section_type.lower(), [])
    
    lines = text.splitlines()
    section_lines = []
    capture = False
    header_found = False
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
  
        is_header = False
        for keyword in keywords:
           
            if line_lower == keyword:
                is_header = True
                break
      
            
            if len(line_stripped) < 50:
             
                if line_lower.startswith(keyword) or re.match(rf'^{re.escape(keyword)}[\s:\-–—]+', line_lower):
                    is_header = True
                    break
                
                if keyword in line_lower and len(line_stripped) < 40:
                   
                    keyword_pos = line_lower.find(keyword)
                    if keyword_pos == 0 or (keyword_pos > 0 and line_stripped[keyword_pos-1] in ' -:'):
                        is_header = True
                        break
        
        if is_header:
            capture = True
            header_found = True
            continue


        if capture and header_found and len(section_lines) > 0:
            should_stop = False
            for stop in stops:
           
                if line_lower.startswith(stop) or re.match(rf'^{re.escape(stop)}[\s:\-–—]+', line_lower):
                    should_stop = True
                    break
             
                if line_lower == stop:
                    should_stop = True
                    break
            
            if should_stop and len(section_lines) >= 2:
                break
            

            if not any(kw in line_lower for kw in keywords):
                section_lines.append(line_stripped)
    

    cleaned = []
    seen = set()
    separator_pattern = re.compile(r'^[-–—=_*#]+$')
    
    for line in section_lines:
        
        if len(line) < 2:
            continue
        
        if separator_pattern.match(line):
            continue
       
        if re.match(r'^(19|20)\d{2}(\s*[-–]\s*(19|20)\d{2})?$', line):
            continue
       
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    
    section_text = "\n".join(cleaned)

    confidence = calculate_section_confidence(section_text, text, section_type.lower())
    
    logger.info(f"Extracted section '{section_type}'. Items found: {len(cleaned)}, Confidence: {confidence}")
    
    return section_text, confidence


def enhance_with_layoutlm(pdf_path, section_type, traditional_result, traditional_confidence):
    if not LAYOUTLM_AVAILABLE:
        logger.info(f"LayoutLMv3 not available for section '{section_type}'")
        return traditional_result, traditional_confidence, False
    
    if traditional_confidence >= LAYOUTLM_CONFIDENCE_THRESHOLD:
        logger.info(f"Traditional extraction confidence ({traditional_confidence}) >= threshold ({LAYOUTLM_CONFIDENCE_THRESHOLD}), skipping LayoutLMv3")
        return traditional_result, traditional_confidence, False
    
    if section_type.lower() not in LAYOUTLM_SECTION_MAP:
        logger.info(f"Section type '{section_type}' not supported by LayoutLMv3")
        return traditional_result, traditional_confidence, False
    
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning(f"Invalid PDF path for LayoutLMv3: {pdf_path}")
        return traditional_result, traditional_confidence, False
    if not str(pdf_path).lower().endswith(".pdf"):
        logger.info(f"LayoutLMv3 skipped for non-PDF file: {pdf_path}")
        return traditional_result, traditional_confidence, False
    
    try:
        logger.info(f"Enhancing section '{section_type}' with LayoutLMv3 (traditional confidence: {traditional_confidence})")
        
        layoutlm_result, layoutlm_confidence = extract_with_layoutlm(pdf_path)
        
        if not layoutlm_result or "error" in layoutlm_result:
            logger.warning(f"LayoutLMv3 extraction failed: {layoutlm_result.get('error', 'Unknown error')}")
            return traditional_result, traditional_confidence, False
        
        layoutlm_section_key = LAYOUTLM_SECTION_MAP.get(section_type.lower())
        layoutlm_value = layoutlm_result.get(layoutlm_section_key)
        
        if not layoutlm_value:
            logger.info(f"LayoutLMv3 did not find section '{section_type}'")
            return traditional_result, traditional_confidence, False
        
        if layoutlm_confidence > traditional_confidence:
            logger.info(f"LayoutLMv3 enhanced '{section_type}' (confidence: {layoutlm_confidence} vs {traditional_confidence})")
            return layoutlm_value, layoutlm_confidence, True
        else:
            logger.info(f"LayoutLMv3 confidence ({layoutlm_confidence}) not better than traditional ({traditional_confidence})")
            return traditional_result, traditional_confidence, False
            
    except Exception as e:
        logger.error(f"Error enhancing with LayoutLMv3: {e}")
        return traditional_result, traditional_confidence, False


def _try_ats_fallback(text, section_type):
    if not ATS_EXTRACTOR_AVAILABLE or not text:
        return None
    
    ats_supported = ['name', 'skills', 'education', 'experience', 'summary']
    if section_type.lower() not in ats_supported:
        return None
    
    try:
        ats_results = extract_all_ats_sections(text)
        ats_result = ats_results.get(section_type.lower())
        
        if ats_result and ats_result[0] and ats_result[1] >= ATS_CONFIDENCE_THRESHOLD:
            logger.info(f"ATS fallback used for '{section_type}' (confidence: {ats_result[1]})")
            return ats_result
        
    except Exception as e:
        logger.debug(f"ATS fallback failed for '{section_type}': {e}")
    
    return None


def _try_ats_primary(text, section_type):
    if not ATS_EXTRACTOR_AVAILABLE or not text:
        return None
    
    if not is_likely_ats_format(text):
        return None
    
    ats_supported = ['name', 'skills', 'education', 'experience', 'summary']
    if section_type.lower() not in ats_supported:
        return None
    
    try:
        ats_results = extract_all_ats_sections(text)
        ats_result = ats_results.get(section_type.lower())
        
        if ats_result and ats_result[0]:
            logger.info(f"ATS primary extraction used for '{section_type}' (confidence: {ats_result[1]})")
            return ats_result
        
    except Exception as e:
        logger.debug(f"ATS primary extraction failed for '{section_type}': {e}")
    
    return None


def extract_section_from_resume(text, section_type, pdf_path=None):
    if not text:
        return "", 0.0
    

    section_type = section_type.lower().strip()
    layout_text, layout_confidence = _get_layout_cued_section(pdf_path, section_type)


    if section_type == "fulltext":
        from src.utils.formatter import clean_fulltext_format
        result = clean_fulltext_format(text)
        return result, 0.95

    if section_type in {"unmatched", "other", "misc"}:
        try:
            parsed = build_intermediate_resume(text)
            unmatched = "\n".join(parsed.unmatched_lines).strip()
            return unmatched, 0.5 if unmatched else 0.0
        except Exception:
            return "", 0.0
    
    if section_type == "name":
        from src.models.name import extract_name_from_resume
        result, confidence = extract_name_from_resume(text, pdf_path)
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        
        
        if confidence < ATS_CONFIDENCE_THRESHOLD:
            ats_result = _try_ats_fallback(text, section_type)
            if ats_result and ats_result[1] > confidence:
                result, confidence = ats_result
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
        )
        
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    

    if section_type == "skills":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)

        result, confidence = extract_skills_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD and not isolated_text:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        
        
        if _allow_ats_fallback(isolated_text, isolated_confidence, confidence):
            ats_result = _try_ats_fallback(text, section_type)
            if ats_result and ats_result[1] > confidence:
                result, confidence = ats_result
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "education":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)

        result, confidence = extract_education_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD and not isolated_text:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        
        
        if _allow_ats_fallback(isolated_text, isolated_confidence, confidence):
            ats_result = _try_ats_fallback(text, section_type)
            if ats_result and ats_result[1] > confidence:
                result, confidence = ats_result
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "certifications":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)

        result, confidence = extract_certifications_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD and not isolated_text:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "projects":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)

        result, confidence = extract_projects_section(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD and not isolated_text:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "experience":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)

        result, confidence = extract_experience_from_resume(scoped_text or text)
        if not isolated_text and not layout_text and confidence < 0.35:
            return "", 0.0
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        
        if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD and not isolated_text:
            result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
        
        
        if _allow_ats_fallback(isolated_text, isolated_confidence, confidence):
            ats_result = _try_ats_fallback(text, section_type)
            if ats_result and ats_result[1] > confidence:
                result, confidence = ats_result
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "awards":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_awards_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "references":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_references_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "declaration":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_declaration_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "contact":
        result, confidence = extract_contact_from_resume(text)
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "strengths":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_strengths_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "training":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_training_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "extra-curricular":
        isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
        if not isolated_text and layout_text:
            isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
        if not isolated_text and section_type in STRICT_HEADING_REQUIRED_SECTIONS:
            return "", 0.0
        scoped_text = _compose_section_scoped_text(section_type, isolated_text)
        result, confidence = extract_extracurricular_from_resume(scoped_text or text)
        result, confidence = _choose_best_result(
            isolated_text,
            isolated_confidence,
            result,
            confidence,
        )
        result, confidence = _prefer_layout_cues(
            result,
            confidence,
            layout_text,
            layout_confidence,
            isolated_text,
            isolated_confidence,
        )
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "email":
        result, confidence = extract_email_from_resume(text)
        
        
        if not result or confidence == 0.0:
            emails, conf = extract_all_emails_from_text(text)
            if emails:
                result = "\n".join(emails)
                confidence = conf
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "phone":
        result, confidence = extract_phone_from_resume(text)
        
        
        if not result or confidence == 0.0:
            phones, conf = extract_all_phones_from_text(text)
            if phones:
                result = "\n".join(phones)
                confidence = conf
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    if section_type == "links":
        result, confidence = extract_links_from_resume(text)
        
        
        if not result or confidence == 0.0:
            links, conf = extract_all_links_from_text(text)
            if links:
                result = "\n".join(links)
                confidence = conf
        
        result, confidence, _ = _apply_corrections(section_type, result, confidence)
        return result, confidence
    
    
    result, confidence = extract_section_by_type(text, section_type)
    isolated_text, isolated_confidence, _ = _get_isolated_section(text, section_type)
    if not isolated_text and layout_text:
        isolated_text, isolated_confidence = layout_text, max(isolated_confidence, layout_confidence)
    if (
        section_type in STRICT_HEADING_REQUIRED_SECTIONS
        and not isolated_text
    ):
        return "", 0.0
    result, confidence = _choose_best_result(
        isolated_text,
        isolated_confidence,
        result,
        confidence,
    )
    if pdf_path and confidence < LAYOUTLM_CONFIDENCE_THRESHOLD:
        result, confidence, _ = enhance_with_layoutlm(pdf_path, section_type, result, confidence)
    result, confidence = _prefer_layout_cues(
        result,
        confidence,
        layout_text,
        layout_confidence,
        isolated_text,
        isolated_confidence,
    )
    
    
    result, confidence, _ = _apply_corrections(section_type, result, confidence)
    return result, confidence


def get_available_sections():
    return list(SECTION_KEYWORDS.keys())

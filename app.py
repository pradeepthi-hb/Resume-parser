import os
import re
import pdfminer
import logging
from typing import Dict, Any, Tuple, Optional, List

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from flask import Flask, render_template, request, jsonify, redirect, url_for


app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.models.models import db, init_database, save_resume, get_resume, get_all_resumes, delete_resume, update_resume, search_resumes
from src.utils.text import extract_text_from_pdf
from src.utils.section_extractor import extract_section_from_resume
from src.models.name import extract_name_from_resume
from src.utils.formatter import clean_fulltext_format
from src.utils.headings import detect_headings
from normalizers.builder_schema_mapper import get_builder_schema_mapper, infer_file_type

try:
    from src.extractors.layoutlm_extractor import extract_with_layoutlm, is_layoutlm_available, LAYOUTLM_AVAILABLE
except ImportError:
    LAYOUTLM_AVAILABLE = False
    extract_with_layoutlm = None
    is_layoutlm_available = lambda: False

try:
    from src.utils.new_sections import (
        extract_languages_from_resume,
        extract_interests_from_resume,
        extract_achievements_from_resume,
        extract_publications_from_resume,
        extract_volunteer_from_resume,
        extract_summary_from_resume
    )
    NEW_SECTIONS_AVAILABLE = True
except ImportError:
    NEW_SECTIONS_AVAILABLE = False
    extract_languages_from_resume = None
    extract_interests_from_resume = None
    extract_achievements_from_resume = None
    extract_publications_from_resume = None
    extract_volunteer_from_resume = None
    extract_summary_from_resume = None

try:
    from src.utils.structured_output import get_structured_output_generator, generate_structured_resume
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUT_AVAILABLE = False
    get_structured_output_generator = None
    generate_structured_resume = None

try:
    from src.utils.performance import (
        get_parallel_extractor,
        get_performance_metrics,
        extract_all_sections_optimized,
        LRUCache
    )
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    get_parallel_extractor = None
    get_performance_metrics = None
    extract_all_sections_optimized = None
    LRUCache = None

try:
    from src.extractors.transformers_extractor import (
        is_transformers_available,
        get_transformer_extractor,
        get_ensemble_scorer
    )
    TRANSFORMERS_AVAILABLE = is_transformers_available()
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    is_transformers_available = lambda: False
    get_transformer_extractor = None
    get_ensemble_scorer = None

try:
    from src.extractors.pdf_layout_improved import extract_full_resume_html, extract_layout_html
    
    def extract_layout_json(pdf_path):
        result = extract_full_resume_html(pdf_path)
        return result.json_output
    PDF_LAYOUT_EXTRACTOR_AVAILABLE = True
except ImportError:
    PDF_LAYOUT_EXTRACTOR_AVAILABLE = False
    extract_full_resume_html = None
    extract_layout_json = None
    extract_layout_html = None

CONTINUOUS_LEARNING_AVAILABLE = True

try:
    from src.extractors.handwriting_extractor import is_handwriting_available
    HANDWRITING_OCR_AVAILABLE = is_handwriting_available()
except ImportError:
    HANDWRITING_OCR_AVAILABLE = False
    is_handwriting_available = None

try:
    from src.training.trainer import get_model_trainer
    MODEL_TRAINING_AVAILABLE = True
except ImportError:
    MODEL_TRAINING_AVAILABLE = False
    get_model_trainer = None

try:
    from src.training.data_preparator import get_data_preparator
    TRAINING_DATA_PREPARATOR_AVAILABLE = True
except ImportError:
    TRAINING_DATA_PREPARATOR_AVAILABLE = False
    get_data_preparator = None


try:
    from src.utils.api_routes import register_routes
    register_routes(app)
    FEATURE_API_ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not register feature API routes: {e}")
    FEATURE_API_ROUTES_AVAILABLE = False


BUILDER_NORMALIZATION_AVAILABLE = True


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resumes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db.init_app(app)


with app.app_context():
    db.create_all()
    logger.info("Database tables created successfully")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def cleanup_orphaned_entries():
    try:
        resumes = get_all_resumes()
        cleaned = 0
        for resume in resumes:
            if resume.original_pdf_path and not os.path.exists(resume.original_pdf_path):
                logger.info(f"Cleaning up orphaned database entry: {resume.filename} (file not found)")
                delete_resume(resume.id)
                cleaned += 1
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned database entries")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")



confidence_scores = {
    "text_confidence": 0.0,
    "skills_confidence": 0.0,
    "education_confidence": 0.0,
    "certifications_confidence": 0.0,
    "projects_confidence": 0.0,
    "name_confidence": 0.0,
    "languages_confidence": 0.0,
    "interests_confidence": 0.0,
    "achievements_confidence": 0.0,
    "publications_confidence": 0.0,
    "volunteer_confidence": 0.0,
    "summary_confidence": 0.0,
    "awards_confidence": 0.0,
    "references_confidence": 0.0
}


def calculate_overall_accuracy() -> float:
    valid_scores = [s for s in confidence_scores.values() if s > 0]
    if not valid_scores:
        return 0.0
    overall = sum(valid_scores) / len(valid_scores) * 100
    return round(overall, 2)


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_native_structured_payload(
    text: str,
    text_confidence: float,
    all_sections: Dict[str, Tuple[str, float]],
) -> Dict[str, Any]:
    sections_copy = dict(all_sections)
    sections_copy["text"] = (text, text_confidence)

    if STRUCTURED_OUTPUT_AVAILABLE:
        return generate_structured_resume(text, sections_copy)

    fallback: Dict[str, Any] = {}
    for section_name, (data, confidence) in all_sections.items():
        fallback[section_name] = {
            "raw_text": data or "",
            "structured_data": {},
            "confidence": confidence,
            "method": "section_extractor",
            "error": None,
        }
    fallback["raw_text"] = {
        "raw_text": text,
        "structured_data": {"word_count": len((text or "").split())},
        "confidence": text_confidence,
        "method": "pdfminer",
        "error": None,
    }
    return fallback


def _build_builder_response(
    native_structured: Dict[str, Any],
    overall_accuracy: float,
    local_resume_id: int,
    filename: str,
) -> Dict[str, Any]:
    mapper = get_builder_schema_mapper()
    return mapper.map_parser_output(
        raw_parser_output=native_structured,
        overall_confidence=overall_accuracy,
        local_resume_id=local_resume_id,
        file_type=infer_file_type(filename),
    )



BOLD_FONT_SIZE_THRESHOLD = 12


def extract_text_with_font_size(pdf_path: str) -> List[Tuple[str, float]]:
    text_elements = []
    
    try:
        for page in extract_pages(pdf_path):
            for element in page:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if not text:
                        continue
                    
                    
                    
                    bbox = element.bbox
                    if bbox:
                        font_size = bbox[3] - bbox[1]  
                        
                        font_size = max(6, min(font_size * 0.8, 18))
                        text_elements.append((text, font_size))
                        
    except Exception as e:
        logger.warning(f"Error extracting text with font size: {e}")
    
    return text_elements


def extract_name_from_bold_text(pdf_path: str) -> Tuple[str, float]:
    if not pdf_path or not os.path.exists(pdf_path):
        return "", 0.0
    
    try:
        text_elements = extract_text_with_font_size(pdf_path)
        
        if not text_elements:
            logger.info("No text elements found with font size information")
            return "", 0.0
        
        
        bold_elements = [(text, fs) for text, fs in text_elements if fs >= BOLD_FONT_SIZE_THRESHOLD]
        
        if not bold_elements:
            logger.info("No bold text found in the document")
            return "", 0.0
        
        
        
        
        first_five_bold_lines = bold_elements[:5]
        
        for text, font_size in first_five_bold_lines:
            
            text = text.strip()
            
            
            if len(text) < 2 or len(text) > 50:
                continue
            
            
            if re.search(r'[\w\.-]+@[\w\.-]+', text):
                continue
            if re.search(r'\d{10,}', text):
                continue
            if re.search(r'\d{4}\s*(to|-)\s*\d{4}', text, re.IGNORECASE):
                continue
            
            
            words = text.split()
            if 2 <= len(words) <= 4:
                
                capitalized_count = sum(1 for w in words if w and len(w) > 0 and w[0].isupper())
                if capitalized_count >= len(words) * 0.7:
                    
                    candidate_name = text.title()
                    
                    
                    from src.models.name import has_name_awareness
                    is_valid, awareness_score = has_name_awareness(candidate_name)
                    
                    if is_valid:
                        confidence = min(0.5 + awareness_score * 0.3, 0.85)
                        logger.info(f"Name extracted from bold text: {candidate_name} (confidence: {confidence}, font_size: {font_size})")
                        return candidate_name, round(confidence, 2)
        
        logger.info("No valid name found in first 5 bold lines")
        return "", 0.0
        
    except Exception as e:
        logger.warning(f"Error extracting name from bold text: {e}")
        return "", 0.0


def _preview_for_log(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_for_correction(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").strip().lower().split())


def _calibrate_confidence(field_name: str, confidence: float, trace_context: str) -> float:
    try:
        from src.training.correction_learning import get_confidence_calibrator
        calibrated = get_confidence_calibrator().calibrate(field_name, confidence)
        if calibrated != confidence:
            logger.info(
                "Confidence calibrated [%s] field=%s raw=%s calibrated=%s",
                trace_context,
                field_name,
                round(confidence, 4),
                round(calibrated, 4),
            )
        return calibrated
    except Exception as calibration_error:
        logger.debug("Confidence calibration skipped for field=%s: %s", field_name, calibration_error)
        return confidence


def apply_corrections_to_field(
    field_name: str,
    extracted_value: str,
    original_confidence: float = 0.0,
    *,
    force_ml: bool = False,
    trace_context: str = "default",
) -> Tuple[str, float]:
    if not extracted_value:
        return extracted_value, original_confidence
    
    current_value = extracted_value
    current_confidence = original_confidence
    
    try:
        from src.utils.correction_storage import get_correction_storage
        storage = get_correction_storage()
        normalized_field = _normalize_for_correction(field_name) or "unknown"
        normalized_lookup = _normalize_for_correction(current_value)
        logger.info(
            "Correction trace [%s] field_raw='%s' field_norm='%s' value_norm='%s' value_raw='%s'",
            trace_context,
            field_name,
            normalized_field,
            normalized_lookup[:180],
            _preview_for_log(current_value),
        )

        best_match = storage.find_best_correction(field_name, current_value)
        if best_match:
            corrected = best_match.get("corrected_value", current_value)
            if corrected != current_value:
                logger.info(
                    "Feedback correction [%s] field=%s current='%s' matched='%s' final='%s' type=%s",
                    trace_context,
                    field_name,
                    _preview_for_log(current_value),
                    _preview_for_log(best_match.get("matched_original")),
                    _preview_for_log(corrected),
                    best_match.get("match_type"),
                )
                current_value = corrected
                current_confidence = max(current_confidence, 0.98)
                logger.info(
                    "Correction trace applied [%s] field_norm='%s' final_norm='%s' confidence=%s",
                    trace_context,
                    normalized_field,
                    _normalize_for_correction(current_value)[:180],
                    round(current_confidence, 4),
                )
        else:
            logger.info(
                "No feedback correction match [%s] field=%s field_norm='%s' key='%s' current='%s'",
                trace_context,
                field_name,
                normalized_field,
                normalized_lookup[:180],
                _preview_for_log(current_value),
            )
    except Exception as e:
        logger.debug(f"Feedback correction error for {field_name}: {e}")

    current_confidence = _calibrate_confidence(field_name, current_confidence, trace_context)
    return current_value, current_confidence


def apply_feedback_corrections_to_results(
    results: Dict[str, Tuple[str, float]]
) -> Dict[str, Tuple[str, float]]:
    corrected_results: Dict[str, Tuple[str, float]] = {}
    logger.info("Final-pass correction pipeline starting for %s fields", len(results))

    for field_name, data in results.items():
        value = ""
        confidence = 0.0
        if isinstance(data, tuple) and len(data) >= 2:
            value, confidence = data[0], data[1]
        else:
            value = data if isinstance(data, str) else ""

        corrected_value, corrected_confidence = apply_corrections_to_field(
            field_name,
            value,
            confidence,
            force_ml=True,
            trace_context="final_pipeline",
        )
        corrected_results[field_name] = (corrected_value, corrected_confidence)
        logger.info(
            "Final-pass correction result field=%s raw_before='%s' raw_after='%s' norm_before='%s' norm_after='%s' conf_before=%s conf_after=%s",
            field_name,
            _preview_for_log(value),
            _preview_for_log(corrected_value),
            _normalize_for_correction(value)[:180],
            _normalize_for_correction(corrected_value)[:180],
            round(confidence, 4),
            round(corrected_confidence, 4),
        )

    return corrected_results


def extract_name_with_filename_fallback(text: str, filename: str, pdf_path: Optional[str] = None) -> Tuple[str, float]:
    try:
        extracted_name, confidence = extract_name_from_resume(text, pdf_path)
        
        
        if extracted_name and confidence > 0:
            logger.info(f"Name extracted from text: {extracted_name} (confidence: {confidence})")
            return extracted_name, confidence
    except Exception as e:
        logger.warning(f"Error extracting name from text: {e}")
    
    
    if pdf_path and os.path.exists(pdf_path):
        try:
            bold_name, bold_confidence = extract_name_from_bold_text(pdf_path)
            if bold_name and bold_confidence > 0:
                logger.info(f"Name extracted from bold text (extra method): {bold_name} (confidence: {bold_confidence})")
                return bold_name, bold_confidence
        except Exception as e:
            logger.warning(f"Error extracting name from bold text: {e}")
    
    
    if filename:
        
        
        name_from_file = os.path.splitext(filename)[0]
        
        
        name_from_file = name_from_file.replace('_', ' ').replace('-', ' ')
        
        
        common_words = ['resume', 'cv', 'curriculum', 'vitae', 'document', 'file']
        name_parts = name_from_file.split()
        name_parts = [part for part in name_parts if part.lower() not in common_words]
        name_from_file = ' '.join(name_parts)
        
        
        name_from_file = name_from_file.title()
        
        if name_from_file and len(name_from_file) > 1:
            
            logger.info(f"Using filename as name fallback: {name_from_file}")
            return name_from_file, 0.3  
    
    
    logger.warning("Could not extract name, returning empty string")
    return "", 0.0


def extract_all_sections(text: str, pdf_path: Optional[str] = None, filename: str = "", skip_name: bool = False) -> Dict[str, Tuple[str, float]]:
    results = {}
    
    
    
    if not skip_name:
        try:
            
            name, conf = extract_name_with_filename_fallback(text, filename, pdf_path)
            results["name"] = (name, conf)
            confidence_scores["name_confidence"] = conf
        except Exception as e:
            logger.error(f"Error extracting name: {e}")
            results["name"] = ("", 0.0)
    else:
        results["name"] = ("", 0.0)
    
    try:
        skills, conf = extract_section_from_resume(text, "skills", pdf_path)
        results["skills"] = (skills, conf)
        confidence_scores["skills_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        results["skills"] = ("", 0.0)
    
    try:
        education, conf = extract_section_from_resume(text, "education", pdf_path)
        results["education"] = (education, conf)
        confidence_scores["education_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting education: {e}")
        results["education"] = ("", 0.0)
    
    try:
        experience, conf = extract_section_from_resume(text, "experience", pdf_path)
        results["experience"] = (experience, conf)
        confidence_scores["experience_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting experience: {e}")
        results["experience"] = ("", 0.0)
    
    try:
        projects, conf = extract_section_from_resume(text, "projects", pdf_path)
        results["projects"] = (projects, conf)
        confidence_scores["projects_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting projects: {e}")
        results["projects"] = ("", 0.0)
    
    try:
        certifications, conf = extract_section_from_resume(text, "certifications", pdf_path)
        results["certifications"] = (certifications, conf)
        confidence_scores["certifications_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting certifications: {e}")
        results["certifications"] = ("", 0.0)
    
    
    try:
        awards, conf = extract_section_from_resume(text, "awards", pdf_path)
        results["awards"] = (awards, conf)
        confidence_scores["awards_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting awards: {e}")
        results["awards"] = ("", 0.0)
    
    
    try:
        references, conf = extract_section_from_resume(text, "references", pdf_path)
        results["references"] = (references, conf)
        confidence_scores["references_confidence"] = conf
    except Exception as e:
        logger.error(f"Error extracting references: {e}")
        results["references"] = ("", 0.0)
    
    
    if NEW_SECTIONS_AVAILABLE:
        new_sections = [
            ("languages", extract_languages_from_resume),
            ("interests", extract_interests_from_resume),
            ("achievements", extract_achievements_from_resume),
            ("publications", extract_publications_from_resume),
            ("volunteer", extract_volunteer_from_resume),
            ("summary", extract_summary_from_resume)
        ]
        
        for section_name, extractor_func in new_sections:
            try:
                result, conf = extractor_func(text)
                results[section_name] = (result, conf)
                confidence_scores[f"{section_name}_confidence"] = conf
            except Exception as e:
                logger.error(f"Error extracting {section_name}: {e}")
                results[section_name] = ("", 0.0)
    
    
    additional_sections = [
        "declaration", "contact", "strengths", "training", 
        "extra-curricular", "email", "phone", "links"
    ]
    
    for section_name in additional_sections:
        try:
            result, conf = extract_section_from_resume(text, section_name, pdf_path)
            results[section_name] = (result, conf)
            confidence_scores[f"{section_name}_confidence"] = conf
        except Exception as e:
            logger.error(f"Error extracting {section_name}: {e}")
            results[section_name] = ("", 0.0)

    results = apply_feedback_corrections_to_results(results)

    for section_name, (_, conf) in results.items():
        confidence_key = f"{section_name}_confidence"
        if confidence_key in confidence_scores:
            confidence_scores[confidence_key] = conf

    return results


NEW_SECTION_NAMES = {"languages", "interests", "achievements", "publications", "volunteer", "summary"}


def extract_section_with_corrections(
    text: str,
    section: str,
    file_path: str,
    filename: str,
    trace_context: str,
) -> Tuple[Optional[str], float]:
    if section == "name":
        result, conf = extract_name_with_filename_fallback(text, filename, file_path)
    elif section in NEW_SECTION_NAMES:
        if not NEW_SECTIONS_AVAILABLE:
            return None, 0.0
        extractors = {
            "languages": extract_languages_from_resume,
            "interests": extract_interests_from_resume,
            "achievements": extract_achievements_from_resume,
            "publications": extract_publications_from_resume,
            "volunteer": extract_volunteer_from_resume,
            "summary": extract_summary_from_resume,
        }
        result, conf = extractors[section](text)
    else:
        result, conf = extract_section_from_resume(text, section, file_path)

    result, conf = apply_corrections_to_field(
        section,
        result,
        conf,
        force_ml=True,
        trace_context=trace_context,
    )
    return result, conf




@app.route("/", methods=["GET", "POST"])
def home():
    name = None
    resume_text = None
    skills_section = None
    extracted_education = None
    selected_section = None
    section = None
    certifications = None
    projects = None
    overall_accuracy = 0.0
    
    
    saved_resume_id = None
    saved_filename = None
    save_success = False
    
    languages_section = None
    interests_section = None
    achievements_section = None
    publications_section = None
    volunteer_section = None
    summary_section = None

    if request.method == "POST":
        file = request.files.get("resume")
        text = ""
        text_confidence = 0.0

        if file and file.filename:
            try:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                
                
                text, text_confidence = extract_text_from_pdf(file_path)
                confidence_scores["text_confidence"] = text_confidence
                logger.info(f"Extracted text length: {len(text)}, Confidence: {text_confidence}")
                
                
                
                original_filename = file.filename if file else ""
                name, name_confidence = extract_name_with_filename_fallback(text, original_filename, file_path)
                confidence_scores["name_confidence"] = name_confidence
                
                
                resume_text = clean_fulltext_format(text)
                
                
                pdf_path_for_layoutlm = file_path
                
                skills_section, skills_confidence = extract_section_from_resume(
                    text, "skills", pdf_path_for_layoutlm
                )
                confidence_scores["skills_confidence"] = skills_confidence
                
                extracted_education, education_confidence = extract_section_from_resume(
                    text, "education", pdf_path_for_layoutlm
                )
                confidence_scores["education_confidence"] = education_confidence
                
                experience_section, experience_confidence = extract_section_from_resume(
                    text, "experience", pdf_path_for_layoutlm
                )
                confidence_scores["experience_confidence"] = experience_confidence
                
                certifications, cert_confidence = extract_section_from_resume(
                    text, "certifications", pdf_path_for_layoutlm
                )
                confidence_scores["certifications_confidence"] = cert_confidence
                
                projects, projects_confidence = extract_section_from_resume(
                    text, "projects", pdf_path_for_layoutlm
                )
                confidence_scores["projects_confidence"] = projects_confidence
                
                
                awards_section = ""
                awards_confidence = 0.0
                try:
                    awards_section, awards_confidence = extract_section_from_resume(text, "awards", pdf_path_for_layoutlm)
                    confidence_scores["awards_confidence"] = awards_confidence
                except Exception as e:
                    logger.error(f"Error extracting awards: {e}")
                
                references_section = ""
                references_confidence = 0.0
                try:
                    references_section, references_confidence = extract_section_from_resume(text, "references", pdf_path_for_layoutlm)
                    confidence_scores["references_confidence"] = references_confidence
                except Exception as e:
                    logger.error(f"Error extracting references: {e}")
                
                
                if NEW_SECTIONS_AVAILABLE:
                    languages_section, languages_conf = extract_languages_from_resume(text)
                    confidence_scores["languages_confidence"] = languages_conf
                    
                    interests_section, interests_conf = extract_interests_from_resume(text)
                    confidence_scores["interests_confidence"] = interests_conf
                    
                    achievements_section, achievements_conf = extract_achievements_from_resume(text)
                    confidence_scores["achievements_confidence"] = achievements_conf
                    
                    publications_section, publications_conf = extract_publications_from_resume(text)
                    confidence_scores["publications_confidence"] = publications_conf
                    
                    volunteer_section, volunteer_conf = extract_volunteer_from_resume(text)
                    confidence_scores["volunteer_confidence"] = volunteer_conf
                    
                    summary_section, summary_conf = extract_summary_from_resume(text)
                    confidence_scores["summary_confidence"] = summary_conf
                
                selected_section = request.form.get("section")
                section = request.form.get("section")
                
                overall_accuracy = calculate_overall_accuracy()
                
                
                try:
                    
                    structured_data = {
                        "name": {"raw_text": name, "confidence": name_confidence},
                        "skills": {"raw_text": skills_section, "confidence": skills_confidence},
                        "education": {"raw_text": extracted_education, "confidence": education_confidence},
                        "experience": {"raw_text": experience_section, "confidence": experience_confidence},
                        "projects": {"raw_text": projects, "confidence": projects_confidence},
                        "certifications": {"raw_text": certifications, "confidence": cert_confidence},
                        "awards": {"raw_text": awards_section, "confidence": awards_confidence},
                        "references": {"raw_text": references_section, "confidence": references_confidence},
                    }
                    
                    
                    if NEW_SECTIONS_AVAILABLE:
                        structured_data["languages"] = {"raw_text": languages_section, "confidence": languages_conf}
                        structured_data["interests"] = {"raw_text": interests_section, "confidence": interests_conf}
                        structured_data["achievements"] = {"raw_text": achievements_section, "confidence": achievements_conf}
                        structured_data["publications"] = {"raw_text": publications_section, "confidence": publications_conf}
                        structured_data["volunteer"] = {"raw_text": volunteer_section, "confidence": volunteer_conf}
                        structured_data["summary"] = {"raw_text": summary_section, "confidence": summary_conf}
                    
                    
                    layout_html = ""
                    if PDF_LAYOUT_EXTRACTOR_AVAILABLE:
                        try:
                            layout_result = extract_full_resume_html(file_path)
                            layout_html = layout_result.html_output
                        except Exception as e:
                            logger.warning(f"Could not extract layout HTML: {e}")
                    
                    
                    resume = save_resume(
                        filename=original_filename,
                        structured_data=structured_data,
                        extracted_text=text,
                        layout_html=layout_html,
                        original_pdf_path=file_path
                    )
                    
                    saved_resume_id = resume.id
                    saved_filename = resume.filename
                    save_success = True
                    logger.info(f"Resume saved to database: ID {resume.id}, Filename: {resume.filename}")
                    
                except Exception as e:
                    logger.error(f"Error saving resume to database: {e}")
                    save_success = False
                
            except Exception as e:
                logger.error(f"Error processing resume: {e}")
                overall_accuracy = 0.0

    return render_template(
        "index.html",
        name=name,
        resume_text=resume_text,
        skills_section=skills_section,
        extracted_education=extracted_education,
        selected_section=selected_section,
        section=section,
        certifications=certifications,
        projects=projects,
        overall_accuracy=overall_accuracy,
        text_confidence=confidence_scores["text_confidence"],
        skills_confidence=confidence_scores["skills_confidence"],
        education_confidence=confidence_scores["education_confidence"],
        certifications_confidence=confidence_scores["certifications_confidence"],
        projects_confidence=confidence_scores["projects_confidence"],
        languages_section=languages_section,
        interests_section=interests_section,
        achievements_section=achievements_section,
        publications_section=publications_section,
        volunteer_section=volunteer_section,
        summary_section=summary_section,
        saved_resume_id=saved_resume_id,
        saved_filename=saved_filename,
        save_success=save_success,
    )


@app.route("/extract", methods=["POST"])
def extract_ajax():
    file = request.files.get("resume")
    section = request.form.get("section")

    if not file or not section:
        return jsonify({"result": "Invalid input"}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        text, text_confidence = extract_text_from_pdf(file_path)
        
        result_data = {"result": "", "confidence": 0.0}

        if section == "fulltext":
            result = clean_fulltext_format(text)
            result_data = {"result": result, "confidence": text_confidence}
        else:
            result, conf = extract_section_with_corrections(
                text=text,
                section=section,
                file_path=file_path,
                filename=file.filename if file else "",
                trace_context="extract_ajax",
            )
            if result is None:
                result_data = {"result": "Section not available", "confidence": 0.0}
            else:
                result_data = {"result": result, "confidence": conf}

        overall_accuracy = calculate_overall_accuracy()
        result_data["overall_accuracy"] = overall_accuracy
        
        
        try:
            from src.models.models import get_all_resumes
            resumes = get_all_resumes()
            if resumes:
                result_data["resume_id"] = resumes[-1].id
        except:
            pass
        
        return jsonify(result_data)
    
    except Exception as e:
        logger.error(f"Error in extract_ajax: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect-headings", methods=["POST"])
def detect_headings_api():
    file = request.files.get("resume")

    if not file or not file.filename:
        return jsonify({
            "status": "error",
            "message": "Resume file is required"
        }), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        text, text_confidence = extract_text_from_pdf(file_path)
        
        if not text or len(text.strip()) < 10:
            ext = os.path.splitext(file.filename or "")[1].lower()
            is_image = ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
            if is_image and not HANDWRITING_OCR_AVAILABLE:
                return jsonify({
                    "status": "error",
                    "message": "OCR engine is unavailable for image resumes. Install/configure Tesseract OCR and restart the app."
                }), 400
            return jsonify({
                "status": "error",
                "message": "Could not extract text from the resume"
            }), 400
        
        headings, headings_confidence = detect_headings(text)
        heading_texts = [heading[1] for heading in headings]
        
        
        try:
            all_sections = extract_all_sections(text, file_path, file.filename, skip_name=True)
            
            
            sections_table = []
            for section_name, (data, confidence) in all_sections.items():
                if data and data.strip():
                    sections_table.append({
                        "section": section_name.replace('-', ' ').title(),
                        "content": data,
                        "confidence": round(confidence * 100, 1)
                    })
            
            structured_data = {}
            if STRUCTURED_OUTPUT_AVAILABLE:
                all_sections["text"] = (text, text_confidence)
                structured_data = generate_structured_resume(text, all_sections)
            else:
                for section_name, (data, confidence) in all_sections.items():
                    structured_data[section_name] = {
                        "raw_text": data,
                        "confidence": confidence
                    }
            
            layout_html = ""
            if PDF_LAYOUT_EXTRACTOR_AVAILABLE:
                try:
                    if str(file.filename).lower().endswith(".pdf"):
                        layout_result = extract_full_resume_html(file_path)
                        layout_html = layout_result.html_output
                except Exception as e:
                    logger.warning(f"Could not extract layout HTML: {e}")
            
            resume = save_resume(
                filename=file.filename,
                structured_data=structured_data,
                extracted_text=text,
                layout_html=layout_html,
                original_pdf_path=file_path
            )
            logger.info(f"Resume saved to database: ID {resume.id}, Filename: {resume.filename}")
            
            return jsonify({
                "status": "success",
                "headings": heading_texts,
                "headings_confidence": headings_confidence,
                "text_confidence": text_confidence,
                "saved_resume_id": resume.id,
                "all_sections": sections_table,
                "full_text": text
            })
        except Exception as e:
            logger.error(f"Error saving resume to database: {e}")
            
            return jsonify({
                "status": "success",
                "headings": heading_texts,
                "headings_confidence": headings_confidence,
                "text_confidence": text_confidence,
                "all_sections": [],
                "full_text": text
            })

    except Exception as e:
        logger.error(f"Error detecting headings: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/parse", methods=["POST"])
def api_parse_resume():
    file = request.files.get("resume")
    section = request.form.get("section")
    response_mode = (request.form.get("response_mode") or request.form.get("output_mode") or "").strip().lower()
    builder_mode_requested = response_mode in {
        "builder",
        "builder_compatible",
        "normalized",
        "builder_normalized",
    }

    if not file or not section:
        return jsonify({
            "status": "error",
            "message": "resume file and section are required"
        }), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        text, text_confidence = extract_text_from_pdf(file_path)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to extract text from the uploaded file"
        }), 500

    try:
        if section == "fulltext":
            result_data = clean_fulltext_format(text)
            section_confidence = text_confidence
        
        elif section == "structured" and not builder_mode_requested:
            
            if STRUCTURED_OUTPUT_AVAILABLE:
                all_sections = extract_all_sections(text, file_path)
                all_sections["text"] = (text, text_confidence)
                structured = generate_structured_resume(text, all_sections)
                return jsonify({
                    "status": "success",
                    "data": structured,
                    "overall_accuracy": calculate_overall_accuracy()
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Structured output not available"
                }), 500
        elif section in {"builder", "builder_compatible"} or (section == "structured" and builder_mode_requested):
            if not BUILDER_NORMALIZATION_AVAILABLE:
                return jsonify({
                    "success": False,
                    "confidence": 0.0,
                    "resume_data": {},
                    "raw_parser_output": {},
                    "parser_metadata": {
                        "normalization_logs": {
                            "builder_compatibility_issues": ["builder normalization not available"]
                        }
                    }
                }), 500

            all_sections = extract_all_sections(text, file_path)
            native_structured = _build_native_structured_payload(text, text_confidence, all_sections)
            overall_accuracy = calculate_overall_accuracy()
            local_resume_id = _safe_int(request.form.get("local_resume_id"), default=1)
            builder_response = _build_builder_response(
                native_structured=native_structured,
                overall_accuracy=overall_accuracy,
                local_resume_id=local_resume_id,
                filename=file.filename or "",
            )
            status_code = 200 if builder_response.get("success") else 422
            return jsonify(builder_response), status_code

        else:
            result, conf = extract_section_with_corrections(
                text=text,
                section=section,
                file_path=file_path,
                filename=file.filename if file else "",
                trace_context="api_parse_resume",
            )
            if result is None:
                return jsonify({
                    "status": "error",
                    "message": f"Section {section} not available"
                }), 500
            result_data = result
            section_confidence = conf

        if not result_data:
            return jsonify({
                "status": "error",
                "message": f"No data found for section: {section}"
            })

        overall_accuracy = calculate_overall_accuracy()

        return jsonify({
            "status": "success",
            "section": section,
            "data": result_data,
            "confidence": section_confidence,
            "text_confidence": text_confidence,
            "overall_accuracy": overall_accuracy
        })

    except Exception as e:
        logger.error(f"Error in api_parse_resume: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/parse-all", methods=["POST"])
def api_parse_all():
    file = request.files.get("resume")
    response_mode = (request.form.get("response_mode") or request.form.get("output_mode") or "").strip().lower()
    builder_mode_requested = response_mode in {
        "builder",
        "builder_compatible",
        "normalized",
        "builder_normalized",
    }

    if not file:
        return jsonify({
            "status": "error",
            "message": "Resume file is required"
        }), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        text, text_confidence = extract_text_from_pdf(file_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to extract text: {str(e)}"
        }), 500

    try:
        
        all_sections = extract_all_sections(text, file_path)
        
        
        overall_accuracy = calculate_overall_accuracy()

        if builder_mode_requested:
            native_structured = _build_native_structured_payload(text, text_confidence, all_sections)
            local_resume_id = _safe_int(request.form.get("local_resume_id"), default=1)
            builder_response = _build_builder_response(
                native_structured=native_structured,
                overall_accuracy=overall_accuracy,
                local_resume_id=local_resume_id,
                filename=file.filename or "",
            )
            status_code = 200 if builder_response.get("success") else 422
            return jsonify(builder_response), status_code
        
        
        response = {
            "status": "success",
            "overall_accuracy": overall_accuracy,
            "text_confidence": text_confidence,
            "sections": {}
        }
        
        for section_name, (data, confidence) in all_sections.items():
            response["sections"][section_name] = {
                "data": data,
                "confidence": confidence
            }
        
        
        if STRUCTURED_OUTPUT_AVAILABLE:
            structured = _build_native_structured_payload(text, text_confidence, all_sections)
            response["structured"] = structured
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in api_parse_all: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500





@app.route("/extract-layoutlm", methods=["POST"])
def extract_layoutlm():
    file = request.files.get("resume")

    if not file:
        return jsonify({           "status": "error",
            "message": "Resume file is required"
        }), 400

    if not LAYOUTLM_AVAILABLE or extract_with_layoutlm is None:
        return jsonify({
            "status": "error",
            "message": "LayoutLMv3 is not available. Please install required dependencies."
        }), 500

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result, confidence = extract_with_layoutlm(file_path)
        
        return jsonify({
            "status": "success",
            "data": result,
            "confidence": confidence,
            "extraction_method": "layoutlmv3"
        })

    except Exception as e:
        logger.error(f"Error extracting with LayoutLMv3: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "success",
        "features": {
            "layoutlm": LAYOUTLM_AVAILABLE,
            "new_sections": NEW_SECTIONS_AVAILABLE,
            "structured_output": STRUCTURED_OUTPUT_AVAILABLE,
            "performance": PERFORMANCE_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE,
            "pdf_layout_extractor": PDF_LAYOUT_EXTRACTOR_AVAILABLE,
            "continuous_learning": CONTINUOUS_LEARNING_AVAILABLE,
            "handwriting_ocr": HANDWRITING_OCR_AVAILABLE,
            "model_training": MODEL_TRAINING_AVAILABLE,
            "training_data_preparator": TRAINING_DATA_PREPARATOR_AVAILABLE,
            "feature_api_routes": FEATURE_API_ROUTES_AVAILABLE,
            "builder_schema_normalization": BUILDER_NORMALIZATION_AVAILABLE
        },
        "supported_sections": [
            "name", "fulltext", "skills", "education", "experience",
            "projects", "certifications", "languages", "interests",
            "achievements", "publications", "volunteer", "summary",
            "structured" if STRUCTURED_OUTPUT_AVAILABLE else None,
            "builder"
        ],
        "new_features": {
            "feedback_pipeline": {
                "description": "Collect and analyze feedback to improve parser logic, section isolation, and training datasets",
                "endpoints": ["/api/feedback", "/api/learning/stats", "/api/learning/pending"]
            },
            "handwriting_recognition": {
                "description": "OCR for handwritten text in resumes",
                "endpoints": ["/api/ocr/status", "/api/ocr/extract", "/api/ocr/detect-handwriting"]
            },
            "custom_model_training": {
                "description": "Train custom ML models with your own data",
                "endpoints": ["/api/model/train", "/api/model/status", "/api/model/list", "/api/model/deploy"]
            }
        }
    })


@app.route("/layoutlm-status", methods=["GET"])
def layoutlm_status():
    return jsonify({
        "status": "success",
        "available": LAYOUTLM_AVAILABLE
    })


@app.route("/api/extract-layout", methods=["POST"])
def api_extract_layout():
    file = request.files.get("resume")
    output_format = request.form.get("format", "json")
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "Resume file is required"}), 400
    if not PDF_LAYOUT_EXTRACTOR_AVAILABLE:
        return jsonify({"status": "error", "message": "PDF Layout Extractor is not available"}), 500
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    try:
        if not file.filename.lower().endswith(".pdf"):
            text, _ = extract_text_from_pdf(file_path)
            plain_text = clean_fulltext_format(text) if text else ""
            fallback_html = f"<pre>{plain_text}</pre>" if plain_text else ""
            return jsonify({
                "status": "success",
                "html_output": fallback_html,
                "plain_text": plain_text,
                "fallback": "non_pdf_plain_text",
                "summary": {"text_elements": 0}
            })

        result = extract_full_resume_html(file_path)
        text, _ = extract_text_from_pdf(file_path)
        plain_text = clean_fulltext_format(text) if text else ""
        if output_format == "html":
            html_output = result.html_output or ""
            if not html_output.strip() and plain_text:
                html_output = f"<pre>{plain_text}</pre>"
            return jsonify({
                "status": "success",
                "html_output": html_output,
                "plain_text": plain_text,
                "extraction_pipeline": ["PDF", "Extract text with positions", "Flow-based HTML reconstruction"],
                "summary": {"text_elements": len(result.text_elements)}
            })
        else:
            return jsonify(result.json_output)
    except Exception as e:
        logger.error(f"Error in layout extraction: {e}")
        try:
            text, _ = extract_text_from_pdf(file_path)
            plain_text = clean_fulltext_format(text) if text else ""
            fallback_html = f"<pre>{plain_text}</pre>" if plain_text else ""
            if plain_text:
                return jsonify({
                    "status": "success",
                    "html_output": fallback_html,
                    "plain_text": plain_text,
                    "fallback": "layout_error_plain_text",
                    "summary": {"text_elements": 0}
                })
        except Exception:
            pass
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/layout-status", methods=["GET"])
def layout_status():
    return jsonify({"status": "success", "available": PDF_LAYOUT_EXTRACTOR_AVAILABLE, "features": {"bounding_boxes": PDF_LAYOUT_EXTRACTOR_AVAILABLE, "table_detection": PDF_LAYOUT_EXTRACTOR_AVAILABLE, "html_reconstruction": PDF_LAYOUT_EXTRACTOR_AVAILABLE, "css_positioning": PDF_LAYOUT_EXTRACTOR_AVAILABLE}})




@app.route("/api/save-resume", methods=["POST"])
def api_save_resume():
    file = request.files.get("resume")
    
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "Resume file is required"}), 400
    
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        
        text, text_confidence = extract_text_from_pdf(file_path)
        
        
        all_sections = extract_all_sections(text, file_path, file.filename)
        
        
        structured_data = {}
        if STRUCTURED_OUTPUT_AVAILABLE:
            all_sections["text"] = (text, text_confidence)
            structured_data = generate_structured_resume(text, all_sections)
        else:
            
            for section_name, (data, confidence) in all_sections.items():
                structured_data[section_name] = {
                    "raw_text": data,
                    "confidence": confidence
                }
        
        
        layout_html = ""
        if PDF_LAYOUT_EXTRACTOR_AVAILABLE:
            try:
                layout_result = extract_full_resume_html(file_path)
                layout_html = layout_result.html_output
            except Exception as e:
                logger.warning(f"Could not extract layout HTML: {e}")
        
        
        resume = save_resume(
            filename=file.filename,
            structured_data=structured_data,
            extracted_text=text,
            layout_html=layout_html,
            original_pdf_path=file_path
        )
        
        return jsonify({
            "status": "success",
            "message": "Resume saved successfully",
            "resume_id": resume.id,
            "filename": resume.filename,
            "sections": list(structured_data.keys()),
            "created_at": resume.created_at.isoformat() if resume.created_at else None
        })
        
    except Exception as e:
        logger.error(f"Error saving resume: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resumes", methods=["GET"])
def api_get_all_resumes():
    try:
        resumes = get_all_resumes()
        return jsonify({
            "status": "success",
            "count": len(resumes),
            "resumes": [r.to_dict() for r in resumes]
        })
    except Exception as e:
        logger.error(f"Error getting resumes: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resume/<int:resume_id>", methods=["GET"])
def api_get_resume(resume_id):
    try:
        resume = get_resume(resume_id)
        if not resume:
            return jsonify({"status": "error", "message": "Resume not found"}), 404
        
        return jsonify({
            "status": "success",
            "resume": resume.to_full_dict()
        })
    except Exception as e:
        logger.error(f"Error getting resume: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resume/<int:resume_id>/preview", methods=["GET"])
def api_get_resume_preview(resume_id):
    try:
        resume = get_resume(resume_id)
        if not resume:
            return jsonify({"status": "error", "message": "Resume not found"}), 404
        
        if not resume.layout_html:
            return jsonify({"status": "error", "message": "No preview available"}), 404
        
        return jsonify({
            "status": "success",
            "resume_id": resume_id,
            "filename": resume.filename,
            "html_output": resume.layout_html,
            "text_elements_count": len(resume.extracted_text) if resume.extracted_text else 0
        })
    except Exception as e:
        logger.error(f"Error getting preview: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resume/<int:resume_id>", methods=["DELETE"])
def api_delete_resume(resume_id):
    try:
        success = delete_resume(resume_id)
        if success:
            return jsonify({"status": "success", "message": "Resume deleted"})
        else:
            return jsonify({"status": "error", "message": "Resume not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting resume: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def api_search_resumes():
    query = request.args.get("q", "")
    if not query:
        return jsonify({"status": "error", "message": "Search query required"}), 400
    
    try:
        resumes = search_resumes(query)
        return jsonify({
            "status": "success",
            "query": query,
            "count": len(resumes),
            "resumes": [r.to_dict() for r in resumes]
        })
    except Exception as e:
        logger.error(f"Error searching resumes: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/preview/<int:resume_id>", methods=["GET"])
def preview_resume(resume_id):
    try:
        resume = get_resume(resume_id)
        if not resume:
            return "Resume not found", 404
        
        return render_template("preview.html", resume=resume)
    except Exception as e:
        logger.error(f"Error previewing resume: {e}")
        return f"Error: {str(e)}", 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


@app.route("/api/delete-file/<filename>", methods=["DELETE"])
def api_delete_file(filename):
    try:
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File deleted from uploads: {filename}")
        
        
        resumes = get_all_resumes()
        deleted = False
        for resume in resumes:
            if resume.filename == filename:
                delete_resume(resume.id)
                logger.info(f"Database entry deleted for: {filename}")
                deleted = True
                break
        
        if deleted:
            return jsonify({
                "status": "success",
                "message": f"File '{filename}' deleted from uploads and database"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "File not found in database"
            }), 404
            
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Enhanced Resume Parser")
    logger.info(f"LayoutLM Available: {LAYOUTLM_AVAILABLE}")
    logger.info(f"New Sections Available: {NEW_SECTIONS_AVAILABLE}")
    logger.info(f"Structured Output Available: {STRUCTURED_OUTPUT_AVAILABLE}")
    logger.info(f"Performance Features Available: {PERFORMANCE_AVAILABLE}")
    logger.info(f"Transformers Available: {TRANSFORMERS_AVAILABLE}")
    logger.info(f"PDF Layout Extractor Available: {PDF_LAYOUT_EXTRACTOR_AVAILABLE}")
    
    
    with app.app_context():
        cleanup_orphaned_entries()
    
    app.run(host="127.0.0.1", port=8001, debug=True)

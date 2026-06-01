import os
import io
import shutil
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)



pytesseract = None
Image = None
cv2 = None
np = None
OCR_AVAILABLE = False
CV2_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OCR dependencies not available: {e}")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenCV dependencies not available: {e}. Preprocessing and handwriting-region detection disabled.")


@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]]
    language: str = "eng"


@dataclass
class HandwritingDetection:
    is_handwritten: bool
    confidence: float
    handwritten_regions: List[Dict[str, Any]]


class HandwritingExtractor:
    
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    IMAGE_DPI = 300
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        lang: str = "eng",
        config: str = "--psm 6 --oem 3"
    ):
        self.tesseract_path = tesseract_path
        self.lang = lang
        self.config = config
        
        if not OCR_AVAILABLE:
            logger.warning("OCR libraries not available. Install pytesseract and pillow")
            return
            
        resolved_path = tesseract_path or self._resolve_tesseract_path()
        if resolved_path:
            pytesseract.pytesseract.tesseract_cmd = resolved_path
    
    @staticmethod
    def _resolve_tesseract_path() -> Optional[str]:
        env_cmd = os.getenv("TESSERACT_CMD")
        if env_cmd and os.path.isfile(env_cmd):
            return env_cmd

        path_cmd = shutil.which("tesseract")
        if path_cmd:
            return path_cmd

        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for cmd in common_paths:
            if os.path.isfile(cmd):
                return cmd
        return None
            
    def is_available(self) -> bool:
        if not OCR_AVAILABLE:
            return False
            
        try:
            
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            return True
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            return False
    
    def preprocess_image(self, image) -> 'np.ndarray':
        if np is None or cv2 is None:
            return image
            
        
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = np.ones((1, 1), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation, kernel, iterations=1)
        
        return img_erode
    
    def detect_handwriting_regions(self, image) -> HandwritingDetection:
        if cv2 is None or np is None or not self.is_available():
            return HandwritingDetection(False, 0.0, [])
            
        try:
            
            if not isinstance(image, np.ndarray):
                image = np.array(image)
                
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            
            
            edges = cv2.Canny(gray, 50, 150)
            
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if 10 < w < 500 and 5 < h < 100:
                    regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    })
            
            
            confidence = min(len(regions) / 10, 1.0) if regions else 0.0
            
            return HandwritingDetection(
                is_handwritten=len(regions) > 0,
                confidence=confidence,
                handwritten_regions=regions
            )
            
        except Exception as e:
            logger.error(f"Error detecting handwriting regions: {e}")
            return HandwritingDetection(False, 0.0, [])
    
    def extract_text(
        self,
        image,
        preprocess: bool = True
    ) -> OCRResult:
        if not self.is_available():
            return OCRResult("", 0.0, [], self.lang)
            
        try:
            
            if isinstance(image, str):
                image = Image.open(image)
                
            
            if preprocess and cv2 is not None:
                processed = self.preprocess_image(image)
            else:
                if np is not None and not isinstance(image, np.ndarray):
                    processed = np.array(image)
                else:
                    processed = image
                    
            
            text = pytesseract.image_to_string(
                processed,
                lang=self.lang,
                config=self.config
            )
            
            
            data = pytesseract.image_to_data(
                processed,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            
            conf_values: List[float] = []
            for conf in data.get("conf", []):
                try:
                    parsed = float(conf)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    conf_values.append(parsed)
            avg_confidence = sum(conf_values) / len(conf_values) / 100.0 if conf_values else 0.0
            
            
            boxes = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                try:
                    word_conf = float(data['conf'][i])
                except (TypeError, ValueError):
                    continue

                if word_conf > 0:
                    boxes.append({
                        "text": data['text'][i],
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                        "confidence": word_conf / 100.0
                    })
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                bounding_boxes=boxes,
                language=self.lang
            )
            
        except Exception as e:
            logger.error(f"Error extracting text with OCR: {e}")
            return OCRResult("", 0.0, [], self.lang)
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[OCRResult]:
        if not self.is_available():
            return []
            
        results = []
        
        try:
            import fitz  
            
            pdf = fitz.open(pdf_path)
            
            for page_num in range(len(pdf)):
                if page_numbers and page_num not in page_numbers:
                    continue
                    
                
                page = pdf[page_num]
                zoom = self.IMAGE_DPI / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                
                detection = self.detect_handwriting_regions(image)
                
                
                result = self.extract_text(image)
                
                
                results.append(result)
                
            pdf.close()
            
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            
        return results
    
    def extract_from_image_file(
        self,
        image_path: str,
        detect_handwriting: bool = True
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "success": False,
                "error": "OCR not available"
            }
            
        try:
            image = Image.open(image_path)
            
            result = {
                "success": True,
                "text": "",
                "confidence": 0.0,
                "handwriting_detected": False,
                "handwriting_regions": []
            }
            
            
            if detect_handwriting:
                detection = self.detect_handwriting_regions(image)
                result["handwriting_detected"] = detection.is_handwritten
                result["handwriting_confidence"] = detection.confidence
                result["handwriting_regions"] = detection.handwritten_regions
                
                
                if detection.is_handwritten and detection.handwritten_regions:
                    handwritten_texts = []
                    for region in detection.handwritten_regions:
                        
                        cropped = image.crop((
                            region["x"],
                            region["y"],
                            region["x"] + region["width"],
                            region["y"] + region["height"]
                        ))
                        
                        
                        region_result = self.extract_text(cropped)
                        handwritten_texts.append(region_result.text)
                        
                    result["text"] = "\n".join(handwritten_texts)
                    result["confidence"] = sum(
                        r.confidence for r in [self.extract_text(image)]
                    ) / 2
                else:
                    
                    ocr_result = self.extract_text(image)
                    result["text"] = ocr_result.text
                    result["confidence"] = ocr_result.confidence
            else:
                
                ocr_result = self.extract_text(image)
                result["text"] = ocr_result.text
                result["confidence"] = ocr_result.confidence
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from image file: {e}")
            return {
                "success": False,
                "error": str(e)
            }



_handwriting_extractor = None


def get_handwriting_extractor(
    tesseract_path: Optional[str] = None,
    lang: str = "eng"
) -> HandwritingExtractor:
    global _handwriting_extractor
    if _handwriting_extractor is None:
        _handwriting_extractor = HandwritingExtractor(tesseract_path, lang)
    return _handwriting_extractor


def is_handwriting_available() -> bool:
    extractor = get_handwriting_extractor()
    return extractor.is_available()


def extract_handwritten_text(
    image_path: str,
    detect_handwriting: bool = True
) -> Dict[str, Any]:
    extractor = get_handwriting_extractor()
    return extractor.extract_from_image_file(image_path, detect_handwriting)

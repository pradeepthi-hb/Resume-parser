import os
import io
import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False
    cv2 = None
    np = None
    Image = None


class ImagePreprocessor:
    
    def __init__(self):
        pass
    
    def is_available(self) -> bool:
        return CV2_AVAILABLE
    
    def load_image(self, image_source) -> Optional[np.ndarray]:
        try:
            if isinstance(image_source, str):
                
                image = cv2.imread(image_source)
                if image is None:
                    
                    pil_image = Image.open(image_source)
                    image = np.array(pil_image)
                    
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            elif isinstance(image_source, Image.Image):
                
                return np.array(image_source)
            elif isinstance(image_source, np.ndarray):
                return image_source
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_threshold(
        self,
        image: np.ndarray,
        method: str = "adaptive"
    ) -> np.ndarray:
        gray = self.to_grayscale(image)
        
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)
        
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)
        
        
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        
        if abs(angle) < 0.5:
            return image
            
        
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def enhance_contrast(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        gray = self.to_grayscale(image)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def remove_borders(
        self,
        image: np.ndarray,
        threshold: int = 200
    ) -> np.ndarray:
        gray = self.to_grayscale(image)
        
        
        coords = cv2.findNonZero(
            cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        )
        
        if coords is None:
            return image
            
        
        x, y, w, h = cv2.boundingRect(coords)
        
        
        return image[y:y+h, x:x+w]
    
    def detect_and_correct_orientation(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        gray = self.to_grayscale(image)
        
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            return image, 0
            
        
        angles = []
        for line in lines[:10]:  
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
            
        avg_angle = np.mean(angles)
        
        
        if abs(avg_angle) > 0.5:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            
            rotated = cv2.warpAffine(
                image, matrix, (w, h),
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated, avg_angle
            
        return image, 0
    
    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        denoise: bool = True,
        deskew: bool = True,
        enhance: bool = True,
        remove_borders: bool = True
    ) -> np.ndarray:
        result = image.copy()
        
        if denoise:
            result = self.remove_noise(result)
            
        if deskew:
            result = self.deskew(result)
            
        if enhance:
            result = self.enhance_contrast(result)
            
        if remove_borders:
            result = self.remove_borders(result)
            
        return result
    
    def resize_image(
        self,
        image: np.ndarray,
        max_width: int = 2000,
        max_height: int = 2000,
        maintain_aspect: bool = True
    ) -> np.ndarray:
        h, w = image.shape[:2]
        
        if maintain_aspect:
            scale = min(max_width / w, max_height / h)
            if scale >= 1:
                return image
                
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)
    
    def detect_text_regions(
        self,
        image: np.ndarray,
        min_area: int = 50
    ) -> List[Dict[str, int]]:
        gray = self.to_grayscale(image)
        
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })
                
        
        regions.sort(key=lambda r: (r["y"], r["x"]))
        
        return regions



_image_preprocessor = None


def get_image_preprocessor() -> ImagePreprocessor:
    global _image_preprocessor
    if _image_preprocessor is None:
        _image_preprocessor = ImagePreprocessor()
    return _image_preprocessor


def preprocess_image_for_ocr(
    image_path: str,
    output_path: Optional[str] = None
) -> Optional[str]:
    preprocessor = get_image_preprocessor()
    
    if not preprocessor.is_available():
        logger.warning("Image preprocessing not available")
        return image_path
        
    try:
        
        image = preprocessor.load_image(image_path)
        if image is None:
            return None
            
        
        processed = preprocessor.preprocess_for_ocr(image)
        
        
        if output_path:
            cv2.imwrite(output_path, processed)
            return output_path
        else:
            
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_processed{ext}"
            cv2.imwrite(output_path, processed)
            return output_path
            
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

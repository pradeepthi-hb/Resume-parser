
import os
import re
import hashlib
import logging
import functools
import time
from typing import Callable, Any, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    ttl: float  


class LRUCache:
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            
            if time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.access_order.remove(key)
                return None
            
            
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: float = 3600):
        with self.lock:
            
            if key in self.cache:
                self.access_order.remove(key)
            
            
            while len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                if oldest in self.cache:
                    del self.cache[oldest]
            
            
            self.cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        return len(self.cache)



text_cache = LRUCache(max_size=50)


def cache_text_extraction(ttl: float = 3600):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(pdf_path: str, *args, **kwargs):
            
            cache_key = f"text_{hashlib.md5(pdf_path.encode()).hexdigest()}"
            
            
            cached = text_cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for text extraction: {pdf_path}")
                return cached
            
            
            result = func(pdf_path, *args, **kwargs)
            
            
            text_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator



section_cache = LRUCache(max_size=100)


def cache_section_extraction(ttl: float = 3600):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(text: str, section_type: str, *args, **kwargs):
            
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            cache_key = f"section_{section_type}_{text_hash}"
            
            
            cached = section_cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for section: {section_type}")
                return cached
            
            
            result = func(text, section_type, *args, **kwargs)
            
            
            section_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator




class LazyLoader:
    
    _loaded = {}
    _lock = threading.Lock()
    
    @classmethod
    def load(cls, name: str, loader: Callable) -> Any:
        with cls._lock:
            if name not in cls._loaded:
                logger.info(f"Lazy loading: {name}")
                cls._loaded[name] = loader()
            return cls._loaded[name]
    
    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._loaded
    
    @classmethod
    def unload(cls, name: str):
        with cls._lock:
            if name in cls._loaded:
                del cls._loaded[name]


def lazy_import_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


def lazy_import_layoutlm():
    try:
        from src.extractors.layoutlm_extractor import extract_with_layoutlm, is_layoutlm_available
        return extract_with_layoutlm, is_layoutlm_available
    except ImportError:
        return None, False


def lazy_import_ocr():
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
        return True
    except ImportError:
        return False



LAZY_LOADERS = {
    "spacy": lazy_import_spacy,
    "layoutlm": lazy_import_layoutlm,
    "ocr": lazy_import_ocr
}


def get_lazy_dependency(name: str) -> Any:
    if name not in LAZY_LOADERS:
        raise ValueError(f"Unknown lazy dependency: {name}")
    return LazyLoader.load(name, LAZY_LOADERS[name])




class ParallelExtractor:
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def extract_sections_parallel(
        self, 
        text: str, 
        section_extractors: Dict[str, Callable],
        pdf_path: Optional[str] = None
    ) -> Dict[str, Tuple[str, float]]:
        results = {}
        futures = {}
        
        
        for section_name, extractor_func in section_extractors.items():
            if pdf_path:
                future = self.executor.submit(extractor_func, text, section_name, pdf_path)
            else:
                future = self.executor.submit(extractor_func, text, section_name)
            futures[future] = section_name
        
        
        for future in as_completed(futures):
            section_name = futures[future]
            try:
                result = future.result()
                results[section_name] = result
            except Exception as e:
                logger.error(f"Error extracting {section_name}: {e}")
                results[section_name] = ("", 0.0)
        
        return results
    
    def shutdown(self):
        self.executor.shutdown(wait=True)



_parallel_extractor = None
_extractor_lock = threading.Lock()


def get_parallel_extractor(max_workers: int = 4) -> ParallelExtractor:
    global _parallel_extractor
    with _extractor_lock:
        if _parallel_extractor is None:
            _parallel_extractor = ParallelExtractor(max_workers=max_workers)
        return _parallel_extractor




class PerformanceMetrics:
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.lock = threading.Lock()
    
    def record(self, name: str, duration: float):
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                "count": len(values),
                "total": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        return {name: self.get_stats(name) for name in self.metrics}
    
    def clear(self):
        with self.lock:
            self.metrics.clear()



_metrics = PerformanceMetrics()


def get_performance_metrics() -> PerformanceMetrics:
    return _metrics


def timeit(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        
        _metrics.record(func.__name__, duration)
        logger.info(f"{func.__name__} took {duration:.4f}s")
        
        return result
    return wrapper




@cache_text_extraction(ttl=1800)  
def extract_text_cached(pdf_path: str) -> Tuple[str, float]:
    from src.utils.text import extract_text_from_pdf
    return extract_text_from_pdf(pdf_path)


@cache_section_extraction(ttl=1800)
def extract_section_cached(text: str, section_type: str, pdf_path: str = None) -> Tuple[str, float]:
    from src.utils.section_extractor import extract_section_from_resume
    return extract_section_from_resume(text, section_type, pdf_path)


def extract_all_sections_optimized(
    text: str, 
    pdf_path: Optional[str] = None,
    use_parallel: bool = True
) -> Dict[str, Tuple[str, float]]:
    
    
    section_extractors = {
        "name": lambda t, s, p: extract_section_cached(t, "name", p),
        "skills": lambda t, s, p: extract_section_cached(t, "skills", p),
        "education": lambda t, s, p: extract_section_cached(t, "education", p),
        "experience": lambda t, s, p: extract_section_cached(t, "experience", p),
        "projects": lambda t, s, p: extract_section_cached(t, "projects", p),
        "certifications": lambda t, s, p: extract_section_cached(t, "certifications", p),
    }
    
    
    try:
        from src.utils.new_sections import (
            extract_languages_from_resume,
            extract_interests_from_resume,
            extract_achievements_from_resume,
            extract_publications_from_resume,
            extract_volunteer_from_resume,
            extract_summary_from_resume
        )
        
        section_extractors.update({
            "languages": lambda t, s, p: extract_languages_from_resume(t),
            "interests": lambda t, s, p: extract_interests_from_resume(t),
            "achievements": lambda t, s, p: extract_achievements_from_resume(t),
            "publications": lambda t, s, p: extract_publications_from_resume(t),
            "volunteer": lambda t, s, p: extract_volunteer_from_resume(t),
            "summary": lambda t, s, p: extract_summary_from_resume(t),
        })
    except ImportError as e:
        logger.warning(f"Could not import new sections: {e}")
    
    if use_parallel:
        
        extractor = get_parallel_extractor()
        results = extractor.extract_sections_parallel(text, section_extractors, pdf_path)
    else:
        
        results = {}
        for section_name, extractor_func in section_extractors.items():
            results[section_name] = extractor_func(text, section_name, pdf_path)
    
    return results


if __name__ == "__main__":
    
    print("Testing cache...")
    text_cache.set("test_key", "test_value", 60)
    print("Cache get:", text_cache.get("test_key"))
    print("Cache size:", text_cache.size())
    
    
    @timeit
    def test_function():
        time.sleep(0.1)
        return "done"
    
    test_function()
    print("Metrics:", get_performance_metrics().get_all_stats())

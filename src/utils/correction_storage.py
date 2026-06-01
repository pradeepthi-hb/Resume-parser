import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").strip().lower().split())


class CorrectionStorage:
    """
    Safe correction storage: deterministic exact normalized matches only.

    This intentionally avoids fuzzy/runtime text replacement behavior.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, str]] = {}
        self._cache_key: Optional[str] = None

    def _load_if_needed(self) -> None:
        from src.training.correction_learning import get_correction_learning_store

        store = get_correction_learning_store()
        try:
            mtime = 0.0
            if store.samples_path:
                import os

                if os.path.exists(store.samples_path):
                    mtime = os.path.getmtime(store.samples_path)
            cache_key = f"{store.samples_path}|{mtime}"
        except Exception:
            cache_key = "fallback"

        if self._cache_key == cache_key and self._cache:
            logger.debug("Safe correction cache reused (key=%s)", cache_key)
            return

        corrections_map = store.build_safe_corrections_map()
        self._cache = corrections_map
        self._cache_key = cache_key
        logger.info(
            "Safe correction cache loaded: fields=%s total_entries=%s",
            len(self._cache),
            sum(len(v) for v in self._cache.values()),
        )

    def invalidate_cache(self) -> None:
        self._cache = {}
        self._cache_key = None
        logger.info("Safe correction cache invalidated")

    def _field_key(self, field_name: str) -> str:
        return _normalize(field_name) or "unknown"

    def get_corrections_dict(self, field_name: str) -> Dict[str, str]:
        self._load_if_needed()
        return self._cache.get(self._field_key(field_name), {})

    def get_all_corrections(self) -> Dict[str, Dict[str, str]]:
        self._load_if_needed()
        return self._cache

    def find_best_correction(self, field_name: str, value: str) -> Optional[Dict[str, str]]:
        if value is None:
            logger.info("Correction lookup skipped: value=None field=%s", field_name)
            return None

        self._load_if_needed()
        normalized_field = self._field_key(field_name)
        field_map = self._cache.get(normalized_field, {})
        normalized_value = _normalize(value)
        logger.info(
            "Correction lookup: field_raw='%s' field_norm='%s' value_norm='%s' field_entries=%s cache_fields=%s cache_key=%s",
            field_name,
            normalized_field,
            normalized_value[:180],
            len(field_map),
            len(self._cache),
            self._cache_key,
        )
        if not normalized_value:
            logger.info("Correction lookup miss: empty normalized value for field_norm='%s'", normalized_field)
            return None

        corrected = field_map.get(normalized_value)
        if not corrected:
            logger.info(
                "Correction lookup miss: no exact normalized key for field_norm='%s' lookup_key='%s'",
                normalized_field,
                normalized_value[:180],
            )
            return None

        if _normalize(corrected) == normalized_value:
            logger.info(
                "Correction lookup skipped: corrected value normalizes to same key for field_norm='%s'",
                normalized_field,
            )
            return None

        logger.info(
            "Correction lookup hit: field_norm='%s' lookup_key='%s' corrected='%s'",
            normalized_field,
            normalized_value[:180],
            corrected[:180],
        )
        return {
            "match_type": "exact_normalized",
            "matched_original": value,
            "corrected_value": corrected,
        }

    def apply_correction(self, field_name: str, value: str) -> Optional[str]:
        match = self.find_best_correction(field_name, value)
        if not match:
            return None
        return match.get("corrected_value")

    def has_corrections(self, field_name: Optional[str] = None) -> bool:
        self._load_if_needed()
        if field_name is None:
            return any(self._cache.values())
        return bool(self._cache.get(self._field_key(field_name), {}))


_correction_storage: Optional[CorrectionStorage] = None


def get_correction_storage() -> CorrectionStorage:
    global _correction_storage
    if _correction_storage is None:
        _correction_storage = CorrectionStorage()
    return _correction_storage


def invalidate_correction_cache() -> None:
    get_correction_storage().invalidate_cache()


def refresh_correction_cache() -> Dict[str, Any]:
    storage = get_correction_storage()
    storage.invalidate_cache()
    data = storage.get_all_corrections()
    return {
        "fields": len(data),
        "entries": sum(len(v) for v in data.values()),
    }

import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFLICT_MIN_SUPPORT = 2
DEFAULT_CONFLICT_MIN_SHARE = 0.6
DEFAULT_CONFLICT_MIN_MARGIN = 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_confidence(value: Any) -> float:
    confidence = _safe_float(value, 0.0)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return round(confidence, 4)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = _clean_text(value).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.strip().lower().split())


def _guess_field(value: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return "unknown"
    return normalized


@dataclass
class CorrectionSample:
    sample_id: str
    resume_id: int
    field_name: str
    original_value: str
    corrected_value: str
    confidence_before: float
    confidence_after: float
    feedback_type: str
    status: str
    source: str
    user_id: Optional[str]
    session_id: Optional[str]
    comment: str
    model_name: str
    model_version: str
    extraction_context: Dict[str, Any]
    timestamp: str


class CorrectionLearningStore:
    DEFAULT_SAMPLES_FILE = "structured_corrections.jsonl"
    DEFAULT_REPORT_FILE = "error_analysis.json"

    def __init__(
        self,
        samples_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        learning_dir = os.path.join(base_dir, "learning_data")
        os.makedirs(learning_dir, exist_ok=True)

        self.learning_dir = learning_dir
        self.samples_path = samples_path or os.path.join(learning_dir, self.DEFAULT_SAMPLES_FILE)
        self.report_path = report_path or os.path.join(learning_dir, self.DEFAULT_REPORT_FILE)
        self._ensure_files()

    def _ensure_files(self) -> None:
        os.makedirs(os.path.dirname(self.samples_path), exist_ok=True)
        if not os.path.exists(self.samples_path):
            with open(self.samples_path, "w", encoding="utf-8") as f:
                f.write("")

        if not os.path.exists(self.report_path):
            self.save_report(
                {
                    "generated_at": _utcnow_iso(),
                    "summary": {},
                    "fields": {},
                }
            )

    def _samples_mtime(self) -> float:
        try:
            if self.samples_path and os.path.exists(self.samples_path):
                return os.path.getmtime(self.samples_path)
        except OSError:
            return 0.0
        return 0.0

    def add_sample(
        self,
        *,
        resume_id: Optional[int],
        field_name: str,
        original_value: str,
        corrected_value: str,
        confidence_before: float = 0.0,
        confidence_after: Optional[float] = None,
        feedback_type: str = "correction",
        status: str = "approved",
        source: str = "ui",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        comment: str = "",
        model_name: str = "",
        model_version: str = "",
        extraction_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        field = _guess_field(_clean_text(field_name))
        original = _clean_text(original_value)
        corrected = _clean_text(corrected_value)
        confidence_before = _clamp_confidence(confidence_before)
        if confidence_after is None:
            confidence_after = (
                1.0
                if normalize_text(original) != normalize_text(corrected) and corrected
                else confidence_before
            )
        confidence_after = _clamp_confidence(confidence_after)

        timestamp = _utcnow_iso()
        digest = hashlib.md5(
            f"{timestamp}|{field}|{original}|{corrected}".encode("utf-8", errors="replace")
        ).hexdigest()[:8]
        sample_id = f"cs_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{digest}"

        sample = CorrectionSample(
            sample_id=sample_id,
            resume_id=int(resume_id or 0),
            field_name=field,
            original_value=original,
            corrected_value=corrected,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            feedback_type=str(feedback_type or "correction").strip().lower(),
            status=str(status or "approved").strip().lower(),
            source=str(source or "ui"),
            user_id=user_id,
            session_id=session_id,
            comment=_clean_text(comment),
            model_name=_clean_text(model_name),
            model_version=_clean_text(model_version),
            extraction_context=extraction_context or {},
            timestamp=timestamp,
        )

        record = sample.__dict__
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        logger.info(
            "Structured correction sample stored: id=%s field=%s status=%s type=%s original_norm='%s' corrected_norm='%s'",
            sample_id,
            field,
            sample.status,
            sample.feedback_type,
            normalize_text(original)[:180],
            normalize_text(corrected)[:180],
        )
        return record

    def add_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stored: List[Dict[str, Any]] = []
        for sample in samples:
            stored.append(
                self.add_sample(
                    resume_id=sample.get("resume_id"),
                    field_name=sample.get("field_name", ""),
                    original_value=sample.get("original_value", ""),
                    corrected_value=sample.get("corrected_value", ""),
                    confidence_before=sample.get("confidence_before", 0.0),
                    confidence_after=sample.get("confidence_after"),
                    feedback_type=sample.get("feedback_type", "correction"),
                    status=sample.get("status", "approved"),
                    source=sample.get("source", "ui"),
                    user_id=sample.get("user_id"),
                    session_id=sample.get("session_id"),
                    comment=sample.get("comment", ""),
                    model_name=sample.get("model_name", ""),
                    model_version=sample.get("model_version", ""),
                    extraction_context=sample.get("extraction_context", {}),
                )
            )
        return stored

    def load_samples(
        self,
        *,
        field_name: Optional[str] = None,
        status: Optional[str] = None,
        feedback_type: Optional[str] = None,
        only_changed: bool = False,
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(self.samples_path):
            return []

        target_field = _guess_field(field_name) if field_name else None
        target_status = str(status).strip().lower() if status else None
        target_feedback_type = str(feedback_type).strip().lower() if feedback_type else None
        samples: List[Dict[str, Any]] = []

        with open(self.samples_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_field = _guess_field(sample.get("field_name", ""))
                sample_status = str(sample.get("status", "approved")).strip().lower()
                sample_feedback_type = str(sample.get("feedback_type", "correction")).strip().lower()

                if target_field and sample_field != target_field:
                    continue
                if target_status and sample_status != target_status:
                    continue
                if target_feedback_type and sample_feedback_type != target_feedback_type:
                    continue
                if only_changed:
                    if normalize_text(sample.get("original_value", "")) == normalize_text(sample.get("corrected_value", "")):
                        continue

                sample["field_name"] = sample_field
                sample["status"] = sample_status
                sample["feedback_type"] = sample_feedback_type
                samples.append(sample)

        return samples

    def update_sample_status(self, sample_id: str, status: str) -> bool:
        if not sample_id or not os.path.exists(self.samples_path):
            return False

        target_status = str(status or "").strip().lower()
        if target_status not in {"pending", "approved", "rejected"}:
            return False

        updated = False
        rewritten: List[str] = []

        with open(self.samples_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                context = sample.get("extraction_context") or {}
                context_feedback_id = context.get("feedback_id")
                context_correction_id = context.get("correction_id")

                if (
                    sample.get("sample_id") == sample_id
                    or context_feedback_id == sample_id
                    or context_correction_id == sample_id
                ):
                    sample["status"] = target_status
                    sample["updated_at"] = _utcnow_iso()
                    updated = True

                rewritten.append(json.dumps(sample, ensure_ascii=True))

        if updated:
            with open(self.samples_path, "w", encoding="utf-8") as f:
                for line in rewritten:
                    f.write(line + "\n")

        return updated

    def get_statistics(self) -> Dict[str, Any]:
        samples = self.load_samples()
        stats = {
            "total_samples": len(samples),
            "approved_samples": 0,
            "pending_samples": 0,
            "rejected_samples": 0,
            "changed_samples": 0,
            "by_field": {},
            "by_feedback_type": {},
            "storage_path": self.samples_path,
        }

        for sample in samples:
            status = sample.get("status", "approved")
            feedback_type = sample.get("feedback_type", "correction")
            field = _guess_field(sample.get("field_name", ""))

            if status == "approved":
                stats["approved_samples"] += 1
            elif status == "pending":
                stats["pending_samples"] += 1
            elif status == "rejected":
                stats["rejected_samples"] += 1

            if normalize_text(sample.get("original_value", "")) != normalize_text(sample.get("corrected_value", "")):
                stats["changed_samples"] += 1

            stats["by_feedback_type"][feedback_type] = stats["by_feedback_type"].get(feedback_type, 0) + 1
            stats["by_field"][field] = stats["by_field"].get(field, 0) + 1

        return stats

    def count_samples(self, *, status: Optional[str] = None, only_changed: bool = False) -> int:
        return len(self.load_samples(status=status, only_changed=only_changed))

    def save_report(self, report: Dict[str, Any]) -> None:
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

    def load_report(self) -> Dict[str, Any]:
        if not os.path.exists(self.report_path):
            return {}
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def build_safe_corrections_map(
        self,
        *,
        min_support: int = DEFAULT_CONFLICT_MIN_SUPPORT,
        min_share: float = DEFAULT_CONFLICT_MIN_SHARE,
        min_margin: int = DEFAULT_CONFLICT_MIN_MARGIN,
    ) -> Dict[str, Dict[str, str]]:
        rows = self.load_samples(status="approved")
        rows.sort(key=lambda sample: str(sample.get("timestamp", "")))

        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for sample in rows:
            field = _guess_field(sample.get("field_name", ""))
            original_norm = normalize_text(sample.get("original_value", ""))
            if not original_norm:
                continue
            grouped[(field, original_norm)].append(sample)

        corrections_map: Dict[str, Dict[str, str]] = {}
        for (field, original_norm), events in grouped.items():
            candidate_events: List[Dict[str, Any]] = []
            for sample in events:
                corrected = str(sample.get("corrected_value", "")).strip()
                corrected_norm = normalize_text(corrected)
                feedback_type = str(sample.get("feedback_type", "correction")).strip().lower()
                status = str(sample.get("status", "approved")).strip().lower()

                is_delete = (
                    status == "rejected"
                    or feedback_type == "rejection"
                    or (feedback_type == "correction" and not corrected_norm)
                )
                if is_delete:
                    candidate_events.clear()
                    break

                if status != "approved" or feedback_type != "correction":
                    continue
                if not corrected_norm or corrected_norm == original_norm:
                    continue

                candidate_events.append(
                    {
                        "corrected": corrected,
                        "corrected_norm": corrected_norm,
                        "timestamp": str(sample.get("timestamp", "")),
                    }
                )

            if not candidate_events:
                continue

            counts: Dict[str, int] = defaultdict(int)
            latest_text_by_norm: Dict[str, str] = {}
            latest_ts_by_norm: Dict[str, str] = {}
            for event in candidate_events:
                norm = event["corrected_norm"]
                counts[norm] += 1
                ts = event.get("timestamp", "")
                if norm not in latest_ts_by_norm or ts >= latest_ts_by_norm[norm]:
                    latest_ts_by_norm[norm] = ts
                    latest_text_by_norm[norm] = event["corrected"]

            ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            top_norm, top_count = ranked[0]
            second_count = ranked[1][1] if len(ranked) > 1 else 0
            total_votes = sum(counts.values())
            top_share = (top_count / total_votes) if total_votes else 0.0
            is_conflict = len(ranked) > 1

            accepted = True
            if is_conflict:
                accepted = (
                    top_count >= max(1, int(min_support))
                    and top_share >= float(min_share)
                    and (top_count - second_count) >= int(min_margin)
                )

            if accepted:
                corrections_map.setdefault(field, {})[original_norm] = latest_text_by_norm[top_norm]
        return corrections_map


class CorrectionPatternMiner:
    def __init__(self, store: Optional[CorrectionLearningStore] = None):
        self.store = store or get_correction_learning_store()

    def analyze(
        self,
        *,
        samples: Optional[List[Dict[str, Any]]] = None,
        low_confidence_threshold: float = 0.65,
        top_pairs: int = 10,
    ) -> Dict[str, Any]:
        rows = samples if samples is not None else self.store.load_samples(status="approved")
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in rows:
            grouped[_guess_field(sample.get("field_name", ""))].append(sample)

        analysis = {
            "generated_at": _utcnow_iso(),
            "summary": {
                "total_samples": len(rows),
                "field_count": len(grouped),
                "low_confidence_threshold": low_confidence_threshold,
            },
            "fields": {},
            "top_problem_fields": [],
        }

        field_rank: List[Tuple[str, float]] = []

        for field, samples_for_field in grouped.items():
            total = len(samples_for_field)
            changed = 0
            low_confidence = 0
            low_confidence_changed = 0
            confidence_sum = 0.0
            pair_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

            for sample in samples_for_field:
                before = _clamp_confidence(sample.get("confidence_before", 0.0))
                confidence_sum += before

                original = str(sample.get("original_value", ""))
                corrected = str(sample.get("corrected_value", ""))
                original_norm = normalize_text(original)
                corrected_norm = normalize_text(corrected)

                is_changed = bool(corrected_norm) and original_norm != corrected_norm
                if is_changed:
                    changed += 1

                if before < low_confidence_threshold:
                    low_confidence += 1
                    if is_changed:
                        low_confidence_changed += 1

                if not is_changed:
                    continue

                key = (original_norm, corrected.strip())
                if key not in pair_data:
                    pair_data[key] = {
                        "original_example": original.strip(),
                        "corrected": corrected.strip(),
                        "original_normalized": original_norm,
                        "count": 0,
                        "confidence_sum": 0.0,
                    }

                pair_data[key]["count"] += 1
                pair_data[key]["confidence_sum"] += before

            common_mistakes = []
            for pair in pair_data.values():
                count = pair["count"]
                avg_before = pair["confidence_sum"] / count if count else 0.0
                common_mistakes.append(
                    {
                        "original": pair["original_example"],
                        "corrected": pair["corrected"],
                        "original_normalized": pair["original_normalized"],
                        "count": count,
                        "avg_confidence_before": round(avg_before, 4),
                    }
                )

            common_mistakes.sort(key=lambda x: (-x["count"], x["avg_confidence_before"]))
            common_mistakes = common_mistakes[:top_pairs]

            avg_confidence = round(confidence_sum / total, 4) if total else 0.0
            error_rate = round(changed / total, 4) if total else 0.0
            low_confidence_error_rate = (
                round(low_confidence_changed / low_confidence, 4) if low_confidence else 0.0
            )

            severity = (error_rate * 0.6 + low_confidence_error_rate * 0.4) * max(1.0, total / 5.0)
            field_rank.append((field, severity))

            analysis["fields"][field] = {
                "total_samples": total,
                "changed_samples": changed,
                "error_rate": error_rate,
                "avg_confidence_before": avg_confidence,
                "low_confidence_samples": low_confidence,
                "low_confidence_error_rate": low_confidence_error_rate,
                "common_mistakes": common_mistakes,
            }

        field_rank.sort(key=lambda item: item[1], reverse=True)
        analysis["top_problem_fields"] = [field for field, _ in field_rank[:5]]
        return analysis

    def run_and_store(
        self,
        *,
        low_confidence_threshold: float = 0.65,
        top_pairs: int = 10,
    ) -> Dict[str, Any]:
        report = self.analyze(
            low_confidence_threshold=low_confidence_threshold,
            top_pairs=top_pairs,
        )
        self.store.save_report(report)
        return report


class ConfidenceCalibrator:
    def __init__(self, store: Optional[CorrectionLearningStore] = None):
        self.store = store or get_correction_learning_store()
        self._cached_profile: Optional[Dict[str, Any]] = None
        self._cached_profile_key: Optional[str] = None

    def build_calibration_profile(self) -> Dict[str, Any]:
        mtime = self.store._samples_mtime()
        cache_key = f"{self.store.samples_path}|{mtime}"
        if self._cached_profile_key == cache_key and self._cached_profile is not None:
            return self._cached_profile

        rows = self.store.load_samples(status="approved")
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        profile: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sample in rows:
            grouped[_guess_field(sample.get("field_name", ""))].append(sample)

        for field, samples in grouped.items():
            for low, high in bins:
                bucket = [
                    sample
                    for sample in samples
                    if low <= _clamp_confidence(sample.get("confidence_before", 0.0)) < high
                ]
                if not bucket:
                    continue
                total = len(bucket)
                changed = 0
                for sample in bucket:
                    o = normalize_text(sample.get("original_value", ""))
                    c = normalize_text(sample.get("corrected_value", ""))
                    if c and c != o:
                        changed += 1
                correctness = 1.0 - (changed / total)
                profile[field][f"{low:.1f}-{high:.1f}"] = {
                    "total": total,
                    "changed": changed,
                    "correctness": round(max(0.0, min(correctness, 1.0)), 4),
                }

        result = {
            "generated_at": _utcnow_iso(),
            "fields": profile,
            "sample_count": len(rows),
        }
        self._cached_profile = result
        self._cached_profile_key = cache_key
        return result

    def calibrate(self, field_name: str, raw_confidence: float) -> float:
        raw = _clamp_confidence(raw_confidence)
        profile = self.build_calibration_profile()
        field = _guess_field(field_name)
        field_profile = profile.get("fields", {}).get(field, {})
        if not field_profile:
            return raw

        for bucket, stats in field_profile.items():
            try:
                low_s, high_s = bucket.split("-")
                low = float(low_s)
                high = float(high_s)
            except (ValueError, TypeError):
                continue
            if low <= raw < high:
                correctness = _clamp_confidence(stats.get("correctness", raw))
                # Blend model confidence with empirical correctness to avoid abrupt jumps.
                calibrated = (raw * 0.55) + (correctness * 0.45)
                return round(max(0.0, min(calibrated, 1.0)), 4)

        return raw


class DisabledCorrectionModelEngine:
    def get_model_status(self) -> Dict[str, Any]:
        return {
            "available": False,
            "reason": "Correction model loading is disabled. Feedback is used for analytics and parser improvements.",
        }

    def apply(self, *, field_name: str, value: str, confidence: float = 0.0, force: bool = False) -> Dict[str, Any]:
        return {
            "applied": False,
            "corrected_value": value,
            "confidence": _clamp_confidence(confidence),
            "reason": "disabled",
            "similarity": None,
            "model_version": None,
        }


class DisabledAutoRetrainer:
    def status(self) -> Dict[str, Any]:
        return {
            "ready": False,
            "triggered": False,
            "reason": "disabled",
            "message": "Auto retraining is disabled for correction replay models.",
        }

    def maybe_retrain(self, *, force: bool = False, deploy: Optional[bool] = None) -> Dict[str, Any]:
        return {
            "triggered": False,
            "forced": force,
            "reason": "disabled",
            "message": "Correction replay retraining is disabled.",
        }


_correction_learning_store: Optional[CorrectionLearningStore] = None
_correction_pattern_miner: Optional[CorrectionPatternMiner] = None
_confidence_calibrator: Optional[ConfidenceCalibrator] = None
_disabled_model_engine: Optional[DisabledCorrectionModelEngine] = None
_disabled_auto_retrainer: Optional[DisabledAutoRetrainer] = None


def get_correction_learning_store() -> CorrectionLearningStore:
    global _correction_learning_store
    if _correction_learning_store is None:
        _correction_learning_store = CorrectionLearningStore()
    return _correction_learning_store


def get_correction_pattern_miner() -> CorrectionPatternMiner:
    global _correction_pattern_miner
    if _correction_pattern_miner is None:
        _correction_pattern_miner = CorrectionPatternMiner(get_correction_learning_store())
    return _correction_pattern_miner


def get_confidence_calibrator() -> ConfidenceCalibrator:
    global _confidence_calibrator
    if _confidence_calibrator is None:
        _confidence_calibrator = ConfidenceCalibrator(get_correction_learning_store())
    return _confidence_calibrator


def get_correction_model_engine() -> DisabledCorrectionModelEngine:
    global _disabled_model_engine
    if _disabled_model_engine is None:
        _disabled_model_engine = DisabledCorrectionModelEngine()
    return _disabled_model_engine


def get_auto_retrainer() -> DisabledAutoRetrainer:
    global _disabled_auto_retrainer
    if _disabled_auto_retrainer is None:
        _disabled_auto_retrainer = DisabledAutoRetrainer()
    return _disabled_auto_retrainer


def get_correction_model_trainer() -> None:
    return None

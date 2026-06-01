import os
import json
import logging
import traceback
from flask import Blueprint, request, jsonify
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


api_features_bp = Blueprint('api_features', __name__)


LOW_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_FEEDBACK_MODE = "full"
BATCH_API_VERSION = "safe-batch-v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _refresh_analysis_summary() -> Dict[str, Any]:
    try:
        from src.training.correction_learning import get_correction_pattern_miner

        analysis = get_correction_pattern_miner().run_and_store()
        return {
            "total_samples": analysis.get("summary", {}).get("total_samples", 0),
            "top_problem_fields": analysis.get("top_problem_fields", []),
        }
    except Exception as learning_error:
        logger.error("Could not refresh correction analysis summary: %s", learning_error)
        return {"total_samples": 0, "top_problem_fields": []}


def _refresh_runtime_correction_cache() -> Dict[str, int]:
    try:
        from src.utils.correction_storage import refresh_correction_cache

        state = refresh_correction_cache()
        logger.info(
            "Runtime correction cache refreshed: fields=%s entries=%s",
            state.get("fields", 0),
            state.get("entries", 0),
        )
        return {
            "fields": int(state.get("fields", 0)),
            "entries": int(state.get("entries", 0)),
        }
    except Exception as cache_error:
        logger.warning("Could not refresh correction cache: %s", cache_error)
        return {"fields": 0, "entries": 0}


@api_features_bp.route("/api/feedback", methods=["POST"])
def submit_feedback():
    stage = "init"
    try:
        stage = "imports"
        from src.utils.feedback_collector import get_feedback_collector
        
        stage = "read_payload"
        data = request.get_json() or {}
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        resume_id = int(data.get("resume_id") or 0)
        field_name = data.get("field_name")
        original_value = data.get("original_value")
        corrected_value = data.get("corrected_value")
        feedback_type = (data.get("feedback_type") or "correction").strip().lower()
        confidence_before = _safe_float(data.get("confidence_before", data.get("confidence", 0.0)), 0.0)
        
        if not all([field_name, original_value]) or corrected_value is None:
            return jsonify({
                "status": "error", 
                "message": "Missing required fields"
            }), 400

        if feedback_type not in {"correction", "confirmation", "rejection"}:
            feedback_type = "correction"

        if feedback_type == "correction" and not str(corrected_value or "").strip():
            return jsonify({
                "status": "error",
                "message": "corrected_value cannot be empty for correction. Use feedback_type='rejection' to unlearn.",
            }), 400

        if feedback_type == "confirmation":
            corrected_value = original_value
        if feedback_type == "rejection":
            corrected_value = corrected_value or ""
        sample_status = data.get("status")
        if not sample_status:
            sample_status = "rejected" if feedback_type == "rejection" else "approved"

        value_changed = str(original_value).strip() != str(corrected_value).strip()
        confidence_after = _safe_float(
            data.get("confidence_after", 1.0 if value_changed and corrected_value else confidence_before),
            confidence_before,
        )
        
        stage = "save_feedback"
        collector = get_feedback_collector()
        
        if feedback_type == "confirmation":
            feedback_id = collector.add_confirmation(
                resume_id=resume_id,
                field_name=field_name,
                value=original_value,
                user_id=data.get("user_id"),
            )
        elif feedback_type == "rejection":
            feedback_id = collector.add_rejection(
                resume_id=resume_id,
                field_name=field_name,
                value=original_value,
                reason=data.get("comment", ""),
                user_id=data.get("user_id"),
            )
        else:
            feedback_id = collector.add_correction(
                resume_id=resume_id,
                field_name=field_name,
                original_value=original_value,
                corrected_value=corrected_value,
                user_id=data.get("user_id"),
                comment=data.get("comment", ""),
            )

        
        stage = "save_sample"
        sample_store = None
        sample = None
        try:
            from src.training.correction_learning import get_correction_learning_store
            sample_store = get_correction_learning_store()
            sample_context = _safe_dict(data.get("extraction_context"))
            sample_context["feedback_id"] = feedback_id

            sample = sample_store.add_sample(
                resume_id=resume_id,
                field_name=field_name,
                original_value=original_value,
                corrected_value=corrected_value,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                feedback_type=feedback_type,
                status=sample_status,
                source=data.get("source", "ui"),
                user_id=data.get("user_id"),
                session_id=data.get("session_id"),
                comment=data.get("comment", ""),
                model_name=data.get("model_name", ""),
                model_version=data.get("model_version", ""),
                extraction_context=sample_context,
            )
            logger.info(
                "Feedback persisted: feedback_id=%s sample_id=%s field='%s' status=%s type=%s",
                feedback_id,
                sample.get("sample_id") if sample else None,
                field_name,
                sample_status,
                feedback_type,
            )
        except Exception as sample_error:
            logger.warning(f"Could not save to correction learning store: {sample_error}")
        
        stage = "post_save_learning"
        analysis_summary = _refresh_analysis_summary()
        
        
        collector.mark_as_processed(feedback_id)
        cache_state = _refresh_runtime_correction_cache()
        
        return jsonify({
            "status": "success",
            "feedback_id": feedback_id,
            "sample_id": sample.get("sample_id") if sample else None,
            "message": "Feedback submitted and saved permanently",
            "analysis_summary": analysis_summary,
            "cache_state": cache_state,
        })
        
    except Exception as e:
        logger.exception("Error submitting feedback")
        tb_lines = traceback.format_exc().splitlines()
        return jsonify({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "error_stage": stage,
            "traceback_tail": tb_lines[-8:],
        }), 500


@api_features_bp.route("/api/feedback/batch", methods=["POST"])
def submit_feedback_batch():
    stage = "init"
    processed_items: List[Dict[str, Any]] = []
    try:
        stage = "imports"
        from src.utils.feedback_collector import get_feedback_collector
        from src.training.correction_learning import (
            get_correction_learning_store,
        )

        stage = "read_payload"
        data = request.get_json() or {}
        corrections = data.get("corrections")
        if not isinstance(corrections, list) or not corrections:
            return jsonify({
                "status": "error",
                "message": "corrections must be a non-empty list",
            }), 400

        stage = "init_services"
        resume_id = int(data.get("resume_id") or 0)
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        source = data.get("source", "ui_batch")
        collector = get_feedback_collector()
        sample_store = get_correction_learning_store()

        stage = "iterate_items"
        for idx, item in enumerate(corrections):
            stage = f"item_{idx}_parse"
            field_name = item.get("field_name")
            original_value = item.get("original_value")
            corrected_value = item.get("corrected_value")
            if not field_name or original_value is None or corrected_value is None:
                processed_items.append({
                    "index": idx,
                    "status": "skipped",
                    "reason": "missing field_name/original_value/corrected_value",
                })
                continue

            normalized_original = str(original_value).strip()
            normalized_corrected = str(corrected_value).strip()
            changed = normalized_original != normalized_corrected

            feedback_type = (item.get("feedback_type") or "").strip().lower()
            if feedback_type not in {"correction", "confirmation", "rejection"}:
                feedback_type = "correction" if changed else "confirmation"

            if feedback_type == "correction" and not normalized_corrected:
                processed_items.append({
                    "index": idx,
                    "status": "skipped",
                    "reason": "empty corrected_value for correction",
                })
                continue

            if feedback_type == "confirmation":
                corrected_value = original_value
            elif feedback_type == "rejection":
                corrected_value = corrected_value or ""
            sample_status = item.get("status")
            if not sample_status:
                sample_status = "rejected" if feedback_type == "rejection" else "approved"

            try:
                stage = f"item_{idx}_save_feedback"
                if feedback_type == "confirmation":
                    feedback_id = collector.add_confirmation(
                        resume_id=resume_id,
                        field_name=field_name,
                        value=original_value,
                        user_id=user_id,
                    )
                elif feedback_type == "rejection":
                    feedback_id = collector.add_rejection(
                        resume_id=resume_id,
                        field_name=field_name,
                        value=original_value,
                        reason=item.get("comment", ""),
                        user_id=user_id,
                    )
                else:
                    feedback_id = collector.add_correction(
                        resume_id=resume_id,
                        field_name=field_name,
                        original_value=original_value,
                        corrected_value=corrected_value,
                        user_id=user_id,
                        comment=item.get("comment", ""),
                    )

                stage = f"item_{idx}_save_sample"
                confidence_before = _safe_float(
                    item.get("confidence_before", item.get("confidence", 0.0)),
                    0.0,
                )
                confidence_after = _safe_float(
                    item.get("confidence_after", 1.0 if changed and corrected_value else confidence_before),
                    confidence_before,
                )

                sample_context = _safe_dict(item.get("extraction_context"))
                sample_context["feedback_id"] = feedback_id

                sample = sample_store.add_sample(
                    resume_id=resume_id,
                    field_name=field_name,
                    original_value=original_value,
                    corrected_value=corrected_value,
                    confidence_before=confidence_before,
                    confidence_after=confidence_after,
                    feedback_type=feedback_type,
                    status=sample_status,
                    source=source,
                    user_id=user_id,
                    session_id=session_id,
                    comment=item.get("comment", ""),
                    model_name=item.get("model_name", data.get("model_name", "")),
                    model_version=item.get("model_version", data.get("model_version", "")),
                    extraction_context=sample_context,
                )
                logger.info(
                    "Batch feedback persisted: feedback_id=%s sample_id=%s field='%s' status=%s type=%s item_index=%s",
                    feedback_id,
                    sample.get("sample_id"),
                    field_name,
                    sample_status,
                    feedback_type,
                    idx,
                )

                stage = f"item_{idx}_mark_processed"
                collector.mark_as_processed(feedback_id)
                processed_items.append({
                    "index": idx,
                    "status": "saved",
                    "field_name": field_name,
                    "feedback_id": feedback_id,
                    "sample_id": sample.get("sample_id"),
                    "feedback_type": feedback_type,
                })
            except Exception as item_error:
                processed_items.append({
                    "index": idx,
                    "status": "skipped",
                    "field_name": field_name,
                    "reason": f"processing_error: {item_error}",
                })
                continue

        stage = "post_items_summary"
        saved_count = sum(1 for item in processed_items if item.get("status") == "saved")
        if saved_count == 0:
            return jsonify({
                "status": "error",
                "message": "No valid corrections were provided",
                "items": processed_items,
                "batch_api_version": BATCH_API_VERSION,
            }), 400

        stage = "post_items_analysis"
        analysis_summary = _refresh_analysis_summary()
        cache_state = _refresh_runtime_correction_cache()

        return jsonify({
            "status": "success",
            "batch_api_version": BATCH_API_VERSION,
            "saved_count": saved_count,
            "items": processed_items,
            "analysis_summary": analysis_summary,
            "cache_state": cache_state,
        })

    except Exception as e:
        logger.exception("Error submitting batch feedback")
        tb_lines = traceback.format_exc().splitlines()
        return jsonify({
            "status": "error",
            "batch_api_version": BATCH_API_VERSION,
            "message": f"{e} @stage={stage}",
            "error_type": type(e).__name__,
            "error_stage": stage,
            "traceback_tail": tb_lines[-8:],
            "items": processed_items,
        }), 500


@api_features_bp.route("/api/feedback/confirm", methods=["POST"])
def confirm_extraction():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        from src.training.correction_learning import (
            get_correction_learning_store,
        )
        
        data = request.get_json() or {}
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        resume_id = int(data.get("resume_id") or 0)
        field_name = data.get("field_name")
        value = data.get("value")
        if not field_name or value is None:
            return jsonify({
                "status": "error",
                "message": "Missing required fields",
            }), 400

        collector = get_feedback_collector()
        feedback_id = collector.add_confirmation(
            resume_id=resume_id,
            field_name=field_name,
            value=value,
            user_id=data.get("user_id"),
        )

        sample_context = _safe_dict(data.get("extraction_context"))
        sample_context["feedback_id"] = feedback_id

        sample = get_correction_learning_store().add_sample(
            resume_id=resume_id,
            field_name=field_name,
            original_value=value,
            corrected_value=value,
            confidence_before=_safe_float(data.get("confidence_before", data.get("confidence", 0.0)), 0.0),
            confidence_after=_safe_float(data.get("confidence_after", data.get("confidence", 0.0)), 0.0),
            feedback_type="confirmation",
            status=data.get("status", "approved"),
            source=data.get("source", "ui"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            comment=data.get("comment", ""),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            extraction_context=sample_context,
        )
        logger.info(
            "Confirmation persisted: feedback_id=%s sample_id=%s field='%s'",
            feedback_id,
            sample.get("sample_id"),
            field_name,
        )
        analysis_summary = _refresh_analysis_summary()
        collector.mark_as_processed(feedback_id)
        cache_state = _refresh_runtime_correction_cache()
        
        return jsonify({
            "status": "success",
            "feedback_id": feedback_id,
            "sample_id": sample.get("sample_id"),
            "analysis_summary": analysis_summary,
            "cache_state": cache_state,
        })
        
    except Exception as e:
        logger.error(f"Error confirming extraction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/feedback/reject", methods=["POST"])
def reject_extraction():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        from src.training.correction_learning import (
            get_correction_learning_store,
        )
        
        data = request.get_json() or {}
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        resume_id = int(data.get("resume_id") or 0)
        field_name = data.get("field_name")
        value = data.get("value")
        if not field_name or value is None:
            return jsonify({
                "status": "error",
                "message": "Missing required fields",
            }), 400
        
        collector = get_feedback_collector()
        feedback_id = collector.add_rejection(
            resume_id=resume_id,
            field_name=field_name,
            value=value,
            reason=data.get("reason", ""),
            user_id=data.get("user_id"),
        )

        sample_context = _safe_dict(data.get("extraction_context"))
        sample_context["feedback_id"] = feedback_id

        sample = get_correction_learning_store().add_sample(
            resume_id=resume_id,
            field_name=field_name,
            original_value=value,
            corrected_value="",
            confidence_before=_safe_float(data.get("confidence_before", data.get("confidence", 0.0)), 0.0),
            confidence_after=0.0,
            feedback_type="rejection",
            status=data.get("status", "rejected"),
            source=data.get("source", "ui"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            comment=data.get("reason", ""),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            extraction_context=sample_context,
        )
        logger.info(
            "Rejection persisted: feedback_id=%s sample_id=%s field='%s'",
            feedback_id,
            sample.get("sample_id"),
            field_name,
        )
        analysis_summary = _refresh_analysis_summary()
        collector.mark_as_processed(feedback_id)
        cache_state = _refresh_runtime_correction_cache()
        
        return jsonify({
            "status": "success",
            "feedback_id": feedback_id,
            "sample_id": sample.get("sample_id"),
            "analysis_summary": analysis_summary,
            "cache_state": cache_state,
        })
        
    except Exception as e:
        logger.error(f"Error rejecting extraction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/stats", methods=["GET"])
def get_learning_stats():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        from src.training.correction_learning import get_correction_learning_store

        structured_stats = get_correction_learning_store().get_statistics()
        feedback_stats = get_feedback_collector().get_statistics()
        analysis_summary = _refresh_analysis_summary()
        
        return jsonify({
            "status": "success",
            "feedback_mode": DEFAULT_FEEDBACK_MODE,
            "statistics": {
                "structured_learning": structured_stats,
                "feedback_storage": feedback_stats,
            },
            "analysis_summary": analysis_summary,
        })
        
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/pending", methods=["GET"])
def get_pending_corrections():
    try:
        from src.training.correction_learning import get_correction_learning_store

        structured_pending = get_correction_learning_store().load_samples(status="pending")
        
        return jsonify({
            "status": "success",
            "count": len(structured_pending),
            "corrections": structured_pending,
        })
        
    except Exception as e:
        logger.error(f"Error getting pending corrections: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/approve/<sample_id>", methods=["POST"])
def approve_correction(sample_id):
    try:
        from src.training.correction_learning import (
            get_correction_learning_store,
        )

        success = get_correction_learning_store().update_sample_status(sample_id, "approved")
        
        if success:
            analysis_summary = _refresh_analysis_summary()
            cache_state = _refresh_runtime_correction_cache()
            return jsonify({
                "status": "success",
                "message": "Correction approved",
                "analysis_summary": analysis_summary,
                "cache_state": cache_state,
            })
        else:
            return jsonify({"status": "error", "message": "Correction not found"}), 404
            
    except Exception as e:
        logger.error(f"Error approving correction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/reject/<sample_id>", methods=["POST"])
def reject_correction(sample_id):
    try:
        from src.training.correction_learning import (
            get_correction_learning_store,
        )
        
        data = request.get_json() or {}
        success = get_correction_learning_store().update_sample_status(sample_id, "rejected")
        
        if success:
            analysis_summary = _refresh_analysis_summary()
            cache_state = _refresh_runtime_correction_cache()
            return jsonify({
                "status": "success",
                "message": "Correction rejected",
                "analysis_summary": analysis_summary,
                "cache_state": cache_state,
            })
        else:
            return jsonify({"status": "error", "message": "Correction not found"}), 404
            
    except Exception as e:
        logger.error(f"Error rejecting correction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/error-analysis", methods=["GET"])
def get_error_analysis():
    try:
        from src.training.correction_learning import (
            get_correction_learning_store,
            get_correction_pattern_miner,
        )

        refresh = _safe_bool(request.args.get("refresh"), False)
        store = get_correction_learning_store()
        miner = get_correction_pattern_miner()

        if refresh:
            report = miner.run_and_store()
        else:
            report = store.load_report() or miner.run_and_store()

        return jsonify({
            "status": "success",
            "analysis": report,
        })

    except Exception as e:
        logger.error("Error getting error analysis: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/backfill", methods=["POST"])
def backfill_structured_learning():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        from src.training.correction_learning import (
            normalize_text,
            get_correction_learning_store,
            get_correction_pattern_miner,
        )

        data = request.get_json() or {}
        include_feedback_files = _safe_bool(data.get("include_feedback_files"), True)

        store = get_correction_learning_store()
        existing = store.load_samples()
        existing_keys = set()
        for sample in existing:
            key = (
                normalize_text(sample.get("field_name", "")),
                normalize_text(sample.get("original_value", "")),
                normalize_text(sample.get("corrected_value", "")),
                int(sample.get("resume_id") or 0),
                sample.get("feedback_type", "correction"),
            )
            existing_keys.add(key)

        inserted = 0
        skipped = 0

        if include_feedback_files:
            collector = get_feedback_collector()
            feedback_dir = collector.storage_path
            if os.path.exists(feedback_dir):
                for filename in os.listdir(feedback_dir):
                    if not filename.endswith(".json"):
                        continue
                    filepath = os.path.join(feedback_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            entry = json.load(f)
                    except Exception:
                        skipped += 1
                        continue

                    feedback_type = str(entry.get("feedback_type", "correction")).strip().lower()
                    if feedback_type not in {"correction", "confirmation", "rejection"}:
                        feedback_type = "correction"

                    original_value = entry.get("original_value", "")
                    corrected_value = entry.get("corrected_value", "")
                    if feedback_type == "confirmation":
                        corrected_value = original_value
                    if feedback_type == "rejection":
                        corrected_value = corrected_value or ""

                    status = entry.get("status")
                    if not status:
                        status = "rejected" if feedback_type == "rejection" else "approved"

                    key = (
                        normalize_text(entry.get("field_name", "")),
                        normalize_text(original_value),
                        normalize_text(corrected_value),
                        int(entry.get("resume_id") or 0),
                        feedback_type,
                    )
                    if key in existing_keys:
                        skipped += 1
                        continue

                    store.add_sample(
                        resume_id=entry.get("resume_id"),
                        field_name=entry.get("field_name", ""),
                        original_value=original_value,
                        corrected_value=corrected_value,
                        confidence_before=_safe_float(entry.get("confidence_before", 0.0), 0.0),
                        confidence_after=_safe_float(entry.get("confidence_after", 1.0 if feedback_type != "rejection" else 0.0), 1.0),
                        feedback_type=feedback_type,
                        status=status,
                        source="backfill_feedback_files",
                        user_id=entry.get("user_id"),
                        session_id=entry.get("session_id"),
                        comment=entry.get("comment", ""),
                        extraction_context={
                            "backfill": True,
                            "feedback_id": filename.replace(".json", ""),
                        },
                    )
                    existing_keys.add(key)
                    inserted += 1

        analysis = get_correction_pattern_miner().run_and_store()
        cache_state = _refresh_runtime_correction_cache()

        return jsonify({
            "status": "success",
            "inserted": inserted,
            "skipped": skipped,
            "analysis_summary": {
                "total_samples": analysis.get("summary", {}).get("total_samples", 0),
                "top_problem_fields": analysis.get("top_problem_fields", []),
            },
            "cache_state": cache_state,
        })

    except Exception as e:
        logger.error("Error running structured learning backfill: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/learning/confidence-calibration", methods=["GET"])
def get_confidence_calibration():
    try:
        from src.training.correction_learning import get_confidence_calibrator

        profile = get_confidence_calibrator().build_calibration_profile()
        return jsonify({
            "status": "success",
            "profile": profile,
        })
    except Exception as e:
        logger.error("Error getting confidence calibration profile: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/model/auto-retrain/status", methods=["GET"])
def auto_retrain_status():
    return jsonify({
        "status": "success",
        "auto_retrain": {
            "ready": False,
            "triggered": False,
            "reason": "disabled",
            "message": "Correction replay auto-retraining is disabled.",
        },
    })


@api_features_bp.route("/api/model/auto-retrain", methods=["POST"])
def run_auto_retrain():
    return jsonify({
        "status": "success",
        "result": {
            "triggered": False,
            "reason": "disabled",
            "message": "Correction replay retraining is disabled.",
        },
    })






@api_features_bp.route("/api/model/train", methods=["POST"])
def train_model():
    try:
        from src.training.trainer import get_model_trainer, TrainingConfig
        
        data = request.get_json() or {}
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        model_type = str(data.get("model_type", "spacy")).strip().lower()
        if model_type in {"correction", "correction-rules", "postprocessor"}:
            return jsonify({
                "status": "error",
                "message": "Correction replay model training is disabled. Use /api/learning/error-analysis for parser-improvement feedback.",
            }), 400
        
        config = TrainingConfig(
            model_type=model_type,
            base_model=data.get("base_model", "en_core_web_sm"),
            epochs=data.get("epochs", 10),
            batch_size=data.get("batch_size", 8),
            learning_rate=data.get("learning_rate", 5e-5),
            output_dir=data.get("output_dir", "./models"),
            field=data.get("field", "general")
        )
        
        trainer = get_model_trainer()
        job_id = trainer.create_training_job(config)
        
        
        success = trainer.run_training_job(job_id)
        
        if success:
            return jsonify({
                "status": "success",
                "job_id": job_id,
                "message": "Training completed"
            })
        else:
            job_status = trainer.get_job_status(job_id)
            return jsonify({
                "status": "error",
                "job_id": job_id,
                "message": job_status.get("error_message", "Training failed")
            }), 500
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/model/status", methods=["GET"])
def model_status():
    try:
        from src.training.trainer import get_model_trainer
        from src.training.data_preparator import get_data_preparator
        from src.training.correction_learning import get_correction_learning_store
        
        trainer = get_model_trainer()
        preparator = get_data_preparator()
        
        
        models = trainer.list_models()
        
        
        try:
            data_stats = preparator.get_dataset_statistics()
        except:
            data_stats = {}

        correction_data_stats = get_correction_learning_store().get_statistics()
        
        return jsonify({
            "status": "success",
            "training_available": True,
            "available_models": models,
            "training_data_stats": data_stats,
            "structured_correction_data_stats": correction_data_stats,
            "features": {
                "spacy_training": True,
                "transformer_training": True,
                "custom_ner": True,
                "feedback_based_training": True,
                "correction_model_training": False,
                "periodic_retraining": False,
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/model/status/<job_id>", methods=["GET"])
def get_training_status(job_id):
    try:
        from src.training.trainer import get_model_trainer
        
        trainer = get_model_trainer()
        status = trainer.get_job_status(job_id)
        
        if status:
            return jsonify({
                "status": "success",
                "job_status": status
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Job not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/model/list", methods=["GET"])
def list_models():
    try:
        from src.training.trainer import get_model_trainer
        
        trainer = get_model_trainer()
        models = trainer.list_models()
        
        return jsonify({
            "status": "success",
            "count": len(models),
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/model/deploy", methods=["POST"])
def deploy_model():
    try:
        from src.training.model_registry import get_model_registry
        
        data = request.get_json() or {}
        
        if not data:
            data = {}

        model_name = data.get("model_name", "resume_model")
        version_id = data.get("version_id")

        if str(model_name).strip().lower() == "correction_postprocessor":
            return jsonify({
                "status": "error",
                "message": "Correction replay model deployment is disabled.",
            }), 400
        
        registry = get_model_registry()
        if not version_id:
            latest = registry.get_latest_version(model_name)
            if latest is None:
                return jsonify({
                    "status": "error",
                    "message": f"No versions found for model '{model_name}'"
                }), 404
            version_id = latest.version_id

        success = registry.deploy_version(model_name, version_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model deployed successfully",
                "model_name": model_name,
                "version_id": version_id,
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to deploy model"
            }), 500
            
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/training/data/stats", methods=["GET"])
def get_training_data_stats():
    try:
        from src.training.data_preparator import get_data_preparator
        
        field = request.args.get("field")
        
        preparator = get_data_preparator()
        stats = preparator.get_dataset_statistics(field)
        
        return jsonify({
            "status": "success",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting training data stats: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500






@api_features_bp.route("/api/ocr/status", methods=["GET"])
def ocr_status():
    try:
        from src.extractors.handwriting_extractor import is_handwriting_available, get_handwriting_extractor
        
        available = is_handwriting_available()
        extractor = get_handwriting_extractor()
        
        tesseract_version = None
        if available:
            try:
                import pytesseract
                tesseract_version = str(pytesseract.get_tesseract_version())
            except:
                pass
        
        return jsonify({
            "status": "success",
            "ocr_available": available,
            "tesseract_version": tesseract_version,
            "features": {
                "handwriting_recognition": available,
                "image_preprocessing": available
            }
        })
        
    except Exception as e:
        logger.error(f"Error checking OCR status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/ocr/extract", methods=["POST"])
def extract_with_ocr():
    try:
        from src.utils.ocr_integrator import get_ocr_integrator
        
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            detect_handwriting = request.form.get('detect_handwriting', 'true').lower() == 'true'
            
            integrator = get_ocr_integrator()
            
            if tmp_path.lower().endswith('.pdf'):
                result = integrator.enhanced_extraction_pipeline(
                    tmp_path, 
                    use_ocr=True
                )
            else:
                result = integrator.preprocess_and_extract(
                    tmp_path,
                    extract_handwriting=detect_handwriting
                )
            
            return jsonify({
                "status": "success",
                "result": result
            })
            
        finally:
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        logger.error(f"Error extracting with OCR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/ocr/detect-handwriting", methods=["POST"])
def detect_handwriting():
    try:
        from src.utils.ocr_integrator import get_ocr_integrator
        
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            integrator = get_ocr_integrator()
            result = integrator.detect_handwritten_sections(tmp_path)
            
            return jsonify({
                "status": "success",
                "result": result
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        logger.error(f"Error detecting handwriting: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500






@api_features_bp.route("/api/feedback/stats", methods=["GET"])
def get_feedback_stats():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        
        collector = get_feedback_collector()
        stats = collector.get_statistics()
        
        return jsonify({
            "status": "success",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_features_bp.route("/api/feedback/export", methods=["GET"])
def export_training_data():
    try:
        from src.utils.feedback_collector import get_feedback_collector
        
        field = request.args.get("field")
        fb_type = request.args.get("type")
        
        feedback_types = [fb_type] if fb_type else None
        
        collector = get_feedback_collector()
        data = collector.export_training_data(field, feedback_types)
        
        return jsonify({
            "status": "success",
            "count": len(data),
            "training_data": data
        })
        
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def register_routes(app):
    app.register_blueprint(api_features_bp)
    logger.info("Feature API routes registered")

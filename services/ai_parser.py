import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict

from schemas.resume_schema_defaults import apply_resume_schema_v1_defaults
from services.ai_response_parser import parse_ai_json_response
from src.utils.resume_schema_validator import validate_resume_schema_v1


logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "resume_parser_prompt.txt"


class AIParserError(RuntimeError):
    pass


def parse_resume_with_ai(text: str, filename: str = "") -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AIParserError("GOOGLE_API_KEY is not configured")

    prompt = _load_prompt()
    raw_response = _call_gemini(
        api_key=api_key,
        model=_required_env("AI_MODEL"),
        prompt=prompt,
        resume_text=text,
        filename=filename,
    )
    parsed = parse_ai_json_response(raw_response)
    resume = apply_resume_schema_v1_defaults(parsed)
    validation = validate_resume_schema_v1(resume)
    if not validation["is_valid"]:
        raise AIParserError("; ".join(validation["errors"]))
    return validation["resume"]


def _load_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise AIParserError(f"Could not load AI parser prompt: {PROMPT_PATH}") from exc


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise AIParserError(f"{name} is not configured")
    return value


def _call_gemini(api_key: str, model: str, prompt: str, resume_text: str, filename: str) -> str:
    model_path = urllib.parse.quote(model, safe="")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent?key={api_key}"
    timeout = int(os.getenv("AI_TIMEOUT_SECONDS", "60"))
    max_chars = int(os.getenv("AI_MAX_INPUT_CHARS", "30000"))

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{prompt}\n\n"
                            f"Filename: {filename or 'unknown'}\n\n"
                            f"Resume text:\n{(resume_text or '')[:max_chars]}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise AIParserError(f"Gemini request failed: HTTP {exc.code} {error_body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise AIParserError(f"Gemini request failed: {exc.reason}") from exc

    try:
        data = json.loads(body)
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        logger.debug("Invalid Gemini response body: %s", body[:1000])
        raise AIParserError("Gemini response shape was not recognized") from exc

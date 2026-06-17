import json
import re
from typing import Any, Dict


class AIResponseParseError(ValueError):
    pass


def parse_ai_json_response(response_text: str) -> Dict[str, Any]:
    cleaned = _strip_markdown_fences(response_text or "").strip()
    json_text = _extract_json_object(cleaned)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise AIResponseParseError(f"AI response did not contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AIResponseParseError("AI response JSON must be an object")
    return parsed


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _extract_json_object(text: str) -> str:
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        raise AIResponseParseError("AI response did not contain a JSON object")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    raise AIResponseParseError("AI response JSON object was incomplete")

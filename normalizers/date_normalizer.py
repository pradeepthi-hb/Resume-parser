import re
from typing import Any, Dict, List, Tuple

from normalizers.text_cleaner import clean_ocr_text, coerce_text


MONTH_ALIASES = {
    "jan": "January",
    "january": "January",
    "1": "January",
    "01": "January",
    "feb": "February",
    "february": "February",
    "2": "February",
    "02": "February",
    "mar": "March",
    "march": "March",
    "3": "March",
    "03": "March",
    "apr": "April",
    "april": "April",
    "4": "April",
    "04": "April",
    "may": "May",
    "5": "May",
    "05": "May",
    "jun": "June",
    "june": "June",
    "6": "June",
    "06": "June",
    "jul": "July",
    "july": "July",
    "7": "July",
    "07": "July",
    "aug": "August",
    "august": "August",
    "8": "August",
    "08": "August",
    "sep": "September",
    "sept": "September",
    "september": "September",
    "9": "September",
    "09": "September",
    "oct": "October",
    "october": "October",
    "10": "October",
    "nov": "November",
    "november": "November",
    "11": "November",
    "dec": "December",
    "december": "December",
    "12": "December",
}

_MONTH_TOKEN_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?|0?[1-9]|1[0-2]"
    r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-/]\s*((?:19|20)\d{2}|\d{2})\b")
_CURRENT_RE = re.compile(r"\b(present|current|ongoing|now|till\s+date)\b", re.IGNORECASE)


def normalize_month(value: Any) -> Tuple[str, bool]:
    token = coerce_text(value).lower().replace(".", "")
    if not token:
        return "", False
    normalized = MONTH_ALIASES.get(token, "")
    return normalized, bool(normalized)


def normalize_year(value: Any) -> Tuple[str, bool]:
    token = clean_ocr_text(value)
    token = re.sub(r"[–—−]", "-", token)
    if not token:
        return "", False
    range_match = _YEAR_RANGE_RE.search(token)
    if range_match:
        start_year = range_match.group(1)
        end_year = range_match.group(2)
        if len(end_year) == 2:
            # Reject YYYY-09 style patterns where second token is likely a month, not a year.
            if int(end_year) <= 12:
                return start_year, True
            end_year = start_year[:2] + end_year
        if re.match(r"^(?:19|20)\d{2}$", end_year):
            return end_year, True
    match = _YEAR_RE.search(token)
    if not match:
        return "", False
    return match.group(0), True


def extract_date_range(text: Any) -> Dict[str, Any]:
    raw = clean_ocr_text(text)
    raw = re.sub(r"[–—−]", "-", raw)
    issues: List[str] = []
    if not raw:
        return {
            "startMonth": "",
            "startYear": "",
            "endMonth": "",
            "endYear": "",
            "current": False,
            "issues": [],
        }

    current = bool(_CURRENT_RE.search(raw))
    month_tokens = [m.group(1) for m in _MONTH_TOKEN_RE.finditer(raw)]
    year_tokens = _YEAR_RE.findall(raw)
    range_match = _YEAR_RANGE_RE.search(raw)

    start_month = ""
    end_month = ""
    start_year = ""
    end_year = ""

    if month_tokens:
        start_month, ok = normalize_month(month_tokens[0])
        if not ok:
            issues.append(f"invalid start month token '{month_tokens[0]}'")
    if len(month_tokens) > 1 and not current:
        end_month, ok = normalize_month(month_tokens[1])
        if not ok:
            issues.append(f"invalid end month token '{month_tokens[1]}'")

    if range_match:
        start_year = range_match.group(1)
        end_raw = range_match.group(2)
        if len(end_raw) == 2:
            if int(end_raw) <= 12:
                # likely YYYY-MM; keep start year only
                end_raw = ""
            else:
                end_raw = start_year[:2] + end_raw
        end_year = end_raw if (end_raw and re.match(r"^(?:19|20)\d{2}$", end_raw)) else ""
    elif year_tokens:
        start_year, _ = normalize_year(year_tokens[0])
        if len(year_tokens) > 1 and not current:
            end_year, _ = normalize_year(year_tokens[1])

    if current:
        end_month = ""
        end_year = ""

    if raw and not (start_year or end_year or start_month or end_month or current):
        issues.append("no parseable date tokens found")

    return {
        "startMonth": start_month,
        "startYear": start_year,
        "endMonth": end_month,
        "endYear": end_year,
        "current": current,
        "issues": issues,
    }


def normalize_achieved_date(text: Any) -> Tuple[str, List[str]]:
    raw = clean_ocr_text(text)
    issues: List[str] = []
    if not raw:
        return "", issues

    month_match = _MONTH_TOKEN_RE.search(raw)
    year_match = _YEAR_RE.search(raw)

    month = ""
    year = ""
    if month_match:
        month, ok = normalize_month(month_match.group(1))
        if not ok:
            issues.append(f"invalid month token '{month_match.group(1)}'")
    if year_match:
        year, _ = normalize_year(year_match.group(0))

    if month and year:
        return f"{month} {year}", issues
    if year:
        return year, issues

    issues.append("unable to normalize achieved date")
    return "", issues

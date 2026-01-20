import json
from langdetect import detect


def detect_lang(text: str):
    """Detect language code like 'en' or 'bn' from a text snippet."""
    try:
        return detect(text)
    except Exception:
        return None

##
def to_record(doc: dict, expected_language: str):
    """
    Convert extracted doc -> final JSONL record, enforce language.
    expected_language: 'bn' or 'en'
    Returns dict or None (if rejected).
    """
    text = (doc.get("title", "") + " " + doc.get("body", ""))[:2000]
    lang = detect_lang(text)

    # enforce expected language
    if expected_language == "bn" and lang != "bn":
        return None
    if expected_language == "en" and lang != "en":
        return None

    body = doc.get("body", "") or ""
    return {
        "title": doc.get("title"),
        "body": body,
        "url": doc.get("url"),
        "date": doc.get("date"),
        "language": expected_language,
        "tokens_count": len(body.split()),
    }


def append_jsonl(path: str, record: dict):
    """Append one JSON object as a line (JSONL)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

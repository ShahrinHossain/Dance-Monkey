import json
from langdetect import detect

def detect_lang(text: str):
    try:
        return detect(text)
    except Exception:
        return None

def to_record(doc: dict, expected_language: str):
    text = (doc.get("title", "") + " " + doc.get("body", ""))[:2000]
    lang = detect_lang(text)

    if expected_language == "bn" and lang != "bn":
        return None
    if expected_language == "en" and lang != "en":
        return None

    record = {
        "title": doc["title"],
        "body": doc["body"],
        "url": doc["url"],
        "date": doc.get("date"),
        "language": expected_language,
        "tokens_count": len(doc["body"].split())
    }
    return record

def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

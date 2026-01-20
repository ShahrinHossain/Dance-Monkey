import time
from tqdm import tqdm
import os
import json
from collections import defaultdict

from crawler.url_discovery import get_sitemap_urls
from crawler.article_extractor import (
    fetch_and_extract,
    fetch_and_extract_prothomalo,
    fetch_and_extract_dhakatribune,
)
from crawler.validate_and_save import to_record, append_jsonl

CRAWL_DELAY = 1.0  # polite delay

# ---- Topic keywords (fallback classifier) ----
TOPIC_KEYWORDS = {
    "sports": ["sport", "match", "tournament", "goal", "cricket", "football", "বিশ্বকাপ", "ম্যাচ", "খেলা"],
    "politics": ["election", "minister", "parliament", "government", "নির্বাচন", "মন্ত্রী", "সরকার"],
    "business": ["bank", "market", "economy", "inflation", "টাকা", "বাজার", "অর্থনীতি"],
    "entertainment": ["film", "music", "celebrity", "নাটক", "সিনেমা", "গান"],
}

GENERIC_URL_TOPICS = {"news", "article", "story", "bangladesh", "national", "latest", "other"}

# ---- URL prefilter rules ----
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".avi", ".mkv",
    ".pdf", ".zip", ".rar",
)
SKIP_SUBSTRINGS = (
    "/photo/", "/photos/",
    "/video/", "/videos/",
    "/gallery/", "/galleries/",
    "/tag/", "/topic/",
    "/author/", "/authors/",
    "/search/",
)

def should_skip_url(url: str) -> bool:
    u = url.lower().split("?", 1)[0]  # ignore query params
    if u.endswith(SKIP_EXTENSIONS):
        return True
    for s in SKIP_SUBSTRINGS:
        if s in u:
            return True
    return False


def load_existing_urls(jsonl_path: str) -> set:
    """Load URLs already saved in a jsonl file."""
    urls = set()
    if not os.path.exists(jsonl_path):
        return urls

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                urls.add(json.loads(line)["url"])
            except Exception:
                continue
    return urls


def topic_from_url(url: str) -> str:
    parts = url.split("/")
    topic = parts[3].lower() if len(parts) > 3 else "other"

    # If the first section is generic, try next segment
    if topic in ("news", "article", "story", "bangladesh"):
        if len(parts) > 4 and parts[4]:
            return parts[4].lower()

    return topic


def detect_topic(title: str, body: str, url: str) -> str:
    """
    Hybrid topic detection:
    1) Use URL-derived topic if it looks meaningful.
    2) Otherwise fallback to keyword matching on title+body.
    """
    url_topic = topic_from_url(url)
    if url_topic not in GENERIC_URL_TOPICS and len(url_topic) > 2:
        return url_topic

    text = f"{title} {body}".lower()

    best_topic = "other"
    best_score = 0
    for topic, kws in TOPIC_KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw.lower() in text:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


def build_for_site(site_base: str, out_path: str, expected_language: str, limit: int, max_urls: int = 5000):
    urls = get_sitemap_urls(site_base)

    stats = {
        "total_urls": len(urls),
        "scanned": 0,
        "prefilter_skips": 0,
        "deduped_skips": 0,
        "fetch_or_extract_failed": 0,
        "lang_reject": 0,
        "saved": 0,
    }

    # Avoid duplicates across runs
    seen = load_existing_urls(out_path)

    # Topic diversity control
    topic_counts = defaultdict(int)
    TARGET_TOPICS = 6
    MAX_PER_TOPIC = max(1, limit // TARGET_TOPICS)

    for url in tqdm(urls[:max_urls], desc=f"{site_base}"):
        if stats["saved"] >= limit:
            break

        stats["scanned"] += 1

        # skip obvious non-article urls before any request
        if should_skip_url(url):
            stats["prefilter_skips"] += 1
            continue

        if url in seen:
            stats["deduped_skips"] += 1
            continue
        seen.add(url)

        try:
            # ✅ FIXED: use exactly one extractor based on domain
            if "prothomalo.com" in url:
                doc = fetch_and_extract_prothomalo(url)
            elif "dhakatribune.com" in url:
                doc = fetch_and_extract_dhakatribune(url)
            else:
                doc = fetch_and_extract(url)

            if not doc:
                stats["fetch_or_extract_failed"] += 1
                time.sleep(CRAWL_DELAY)
                continue

            rec = to_record(doc, expected_language)
            if not rec:
                stats["lang_reject"] += 1
                time.sleep(CRAWL_DELAY)
                continue

            # enforce topic diversity before saving
            t = detect_topic(doc.get("title", ""), doc.get("body", ""), doc.get("url", url))
            if topic_counts[t] >= MAX_PER_TOPIC:
                time.sleep(CRAWL_DELAY)
                continue

            append_jsonl(out_path, rec)
            stats["saved"] += 1
            topic_counts[t] += 1

        except Exception:
            stats["fetch_or_extract_failed"] += 1

        time.sleep(CRAWL_DELAY)

    print(f"\nSTATS for {site_base} ({expected_language}): {stats}\n")
    return stats["saved"]


def main():
    # You can add prothomalo back if you want now that extractor routing is fixed
    bn_sites = [
        "https://bangla.bdnews24.com",
        # "https://www.prothomalo.com",
        # "https://banglatribune.com",
        # "https://www.dhakapost.com",
    ]

    en_sites = [
        "https://www.dhakatribune.com",
        # "https://www.thedailystar.net",
        # "https://www.newagebd.net",
        # "https://www.dailynewnation.com",
        # "https://www.daily-sun.com",
    ]

    bn_out = "data/processed/bn.jsonl"
    en_out = "data/processed/en.jsonl"

    bn_target = 2500
    en_target = 2500

    quota_per_site = 800

    total_bn = 0
    for s in bn_sites:
        total_bn += build_for_site(s, bn_out, "bn", quota_per_site)
        if total_bn >= bn_target:
            break

    total_en = 0
    for s in en_sites:
        total_en += build_for_site(s, en_out, "en", quota_per_site)
        if total_en >= en_target:
            break

    print("DONE")
    print("Bangla docs:", total_bn)
    print("English docs:", total_en)


if __name__ == "__main__":
    main()

import time
import json
from tqdm import tqdm
from backend.crawler.url_discovery import get_sitemap_urls
from backend.crawler.article_extractor import fetch_and_extract
from backend.crawler.validate_and_save import to_record, append_jsonl

CRAWL_DELAY = 1.0  # polite delay

def build_for_site(site_base: str, out_path: str, expected_language: str, limit: int):
    urls = get_sitemap_urls(site_base)

    kept = 0
    seen = set()

    for url in tqdm(urls, desc=f"{site_base}"):
        if kept >= limit:
            break
        if url in seen:
            continue
        seen.add(url)

        try:
            doc = fetch_and_extract(url)
            if not doc:
                continue

            rec = to_record(doc, expected_language)
            if not rec:
                continue

            append_jsonl(out_path, rec)
            kept += 1
            time.sleep(CRAWL_DELAY)
        except Exception:
            continue

    return kept

def main():
    # You should load these from yaml later; hardcoding is OK for first run.
    bn_sites = [
        "https://www.prothomalo.com",
        "https://bangla.bdnews24.com",
        "https://www.kalerkantho.com",
        "https://banglatribune.com",
        "https://www.dhakapost.com",
    ]
    en_sites = [
        "https://www.thedailystar.net",
        "https://www.newagebd.net",
        "https://www.dailynewnation.com",
        "https://www.daily-sun.com",
        "https://www.dhakatribune.com",
    ]

    bn_out = "data/processed/bn.jsonl"
    en_out = "data/processed/en.jsonl"

    bn_target = 2500
    en_target = 2500

    # Distribute quota across 5 sites (500 each). You can adjust dynamically.
    quota_per_site = 10

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

import re
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {"User-Agent": "CLIR-Assignment-Crawler/1.0"}

def get_sitemap_urls(site_base: str):
    """
    Tries common sitemap locations and returns a deduplicated list of URLs.
    """
    candidates = [
        urljoin(site_base, "/sitemap.xml"),
        urljoin(site_base, "/sitemap_index.xml"),
        urljoin(site_base, "/sitemap-news.xml"),
    ]
    urls = set()

    for sm_url in candidates:
        try:
            r = requests.get(sm_url, headers=HEADERS, timeout=15)
            if r.status_code != 200 or "xml" not in r.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(r.text, "xml")
            locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
            for u in locs:
                urls.add(u)
        except Exception:
            pass

    # If it's a sitemap index, it may contain other sitemaps
    expanded = set()
    for u in list(urls):
        if u.endswith(".xml") and "sitemap" in u:
            try:
                r = requests.get(u, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "xml")
                    for loc in soup.find_all("loc"):
                        expanded.add(loc.get_text(strip=True))
            except Exception:
                pass

    urls |= expanded

    # Filter obvious non-article URLs lightly (keep generous)
    urls = {u for u in urls if u.startswith("http")}
    return sorted(urls)

def get_rss_urls(rss_url: str):
    feed = feedparser.parse(rss_url)
    return [entry.link for entry in feed.entries if hasattr(entry, "link")]

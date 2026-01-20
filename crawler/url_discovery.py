import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import gzip
import io

HEADERS = {"User-Agent": "CLIR-Assignment-Crawler/1.0"}

def _fetch(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None

def _extract_sitemaps_from_robots(site_base: str):
    robots_url = urljoin(site_base, "/robots.txt")
    content = _fetch(robots_url)
    if not content:
        return []
    lines = content.decode("utf-8", errors="ignore").splitlines()
    sitemaps = []
    for line in lines:
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":", 1)[1].strip())
    return sitemaps

def _parse_sitemap_xml(xml_text: str):
    soup = BeautifulSoup(xml_text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]

def _read_sitemap(url: str):
    raw = _fetch(url)
    if not raw:
        return []

    # Handle .gz compressed sitemap
    if url.endswith(".gz"):
        try:
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        except Exception:
            return []

    xml_text = raw.decode("utf-8", errors="ignore")
    return _parse_sitemap_xml(xml_text)

def get_sitemap_urls(site_base: str, max_nested: int = 50):
    candidates = [
        urljoin(site_base, "/sitemap.xml"),
        urljoin(site_base, "/sitemap_index.xml"),
        urljoin(site_base, "/sitemapindex.xml"),
    ]
    candidates += _extract_sitemaps_from_robots(site_base)

    seen_sitemaps = set()
    seen_urls = set()
    urls = []  # keep order

    queue = [c for c in candidates if c]

    while queue and max_nested > 0:
        sm = queue.pop(0)
        if sm in seen_sitemaps:
            continue
        seen_sitemaps.add(sm)

        locs = _read_sitemap(sm)

        for loc in locs:
            if "sitemap" in loc:
                if loc not in seen_sitemaps:
                    queue.append(loc)
            elif loc.startswith("http"):
                if loc not in seen_urls:
                    seen_urls.add(loc)
                    urls.append(loc)

        max_nested -= 1

    return urls

import re
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

HEADERS = {"User-Agent": "CLIR-Assignment-Crawler/1.0"}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_date(soup: BeautifulSoup):
    candidates = []
    for key in ["article:published_time", "pubdate", "publishdate", "date", "DC.date.issued"]:
        tag = soup.find("meta", {"property": key}) or soup.find("meta", {"name": key})
        if tag and tag.get("content"):
            candidates.append(tag["content"])

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        candidates.append(time_tag["datetime"])

    for c in candidates:
        try:
            return dateparser.parse(c).isoformat()
        except Exception:
            continue
    return None

def extract_title(soup: BeautifulSoup):
    if soup.title and soup.title.text:
        return clean_text(soup.title.text)
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" "))
    return None

def extract_body(soup: BeautifulSoup):
    ps = soup.find_all("p")
    text = " ".join([p.get_text(" ") for p in ps])
    text = clean_text(text)

    # Reject very short / junk pages
    return text if len(text.split()) >= 80 else None


def fetch_and_extract(url: str):
    r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)

    # ---- ADD THIS (hard 404) ----
    if r.status_code == 404:
        return None

    html_lower = r.text.lower()

    # ---- ADD THIS (soft 404 detection for Daily Star) ----
    if "the requested page could not be found" in html_lower:
        return None
    if "<title>page not found" in html_lower:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    title = extract_title(soup)
    body = extract_body(soup)
    date = extract_date(soup)

    if not title or not body:
        return None

    return {
        "url": url,
        "title": title,
        "body": body,
        "date": date,
    }

def fetch_and_extract_dhakatribune(url: str):
    """
    Dhaka Tribune specific extractor.
    Uses site-specific selectors for higher yield.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Title
        h1 = soup.find("h1")
        if not h1:
            return None
        title = clean_text(h1.get_text(" "))

        # Date
        date = None
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            try:
                date = dateparser.parse(time_tag["datetime"]).isoformat()
            except Exception:
                date = None

        # Body (site-specific)
        body_div = soup.find("div", class_="detail_inner")
        if not body_div:
            return None

        body = clean_text(body_div.get_text(" "))
        if len(body.split()) < 80:
            return None

        return {
            "url": url,
            "title": title,
            "body": body,
            "date": date,
        }

    except Exception:
        return None


def fetch_and_extract_prothomalo(url: str):
    """
    Prothom Alo specific extractor.
    Uses site-specific selectors for higher yield.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Title
        h1 = soup.find("h1")
        if not h1:
            return None
        title = clean_text(h1.get_text(" "))

        # Date
        date = None
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            try:
                date = dateparser.parse(time_tag["datetime"]).isoformat()
            except Exception:
                date = None

        # Body (site-specific)
        paragraphs = soup.select(".story-element-text p")
        body = " ".join(p.get_text(" ") for p in paragraphs)
        body = clean_text(body)

        if len(body.split()) < 80:
            return None

        return {
            "url": url,
            "title": title,
            "body": body,
            "date": date,
        }

    except Exception:
        return None
    
    


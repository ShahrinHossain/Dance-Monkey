import re
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

HEADERS = {"User-Agent": "CLIR-Assignment-Crawler/1.0"}

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_date(soup: BeautifulSoup):
    # Try common meta tags
    candidates = []
    for key in ["article:published_time", "pubdate", "publishdate", "date", "DC.date.issued"]:
        tag = soup.find("meta", {"property": key}) or soup.find("meta", {"name": key})
        if tag and tag.get("content"):
            candidates.append(tag["content"])

    # Try <time datetime="...">
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
    """
    Very simple baseline: join all <p> tags.
    Later you can add site-specific rules if needed.
    """
    ps = soup.find_all("p")
    text = " ".join([p.get_text(" ") for p in ps])
    text = clean_text(text)
    # Avoid super-short pages
    return text if len(text) > 200 else None

def fetch_and_extract(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    title = extract_title(soup)
    body = extract_body(soup)
    date = extract_date(soup)

    if not title or not body:
        return None

    return {"url": url, "title": title, "body": body, "date": date}

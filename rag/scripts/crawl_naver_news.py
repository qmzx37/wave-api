from __future__ import annotations
import os, json, time, re, hashlib
from datetime import datetime
from urllib.parse import quote
import requests

OUT = r"C:\llm\train\wave\rag\storage\raw\naver_news.jsonl"
QUERIES = ["동기부여","우울","불안","공부","습관","성장","집중","번아웃"]
MAX_ITEMS = 200
SORT = "date"  # date|sim

CID = os.environ.get("NAVER_CLIENT_ID","").strip()
CSEC = os.environ.get("NAVER_CLIENT_SECRET","").strip()
if not CID or not CSEC:
    raise SystemExit("❌ NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요해.")

def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

headers = {"X-Naver-Client-Id": CID, "X-Naver-Client-Secret": CSEC}

os.makedirs(os.path.dirname(OUT), exist_ok=True)

seen = set()
written = 0

with open(OUT, "a", encoding="utf-8") as f:
    for q in QUERIES:
        start = 1
        display = 100  # naver max
        while start <= 1000 and written < MAX_ITEMS:
            url = f"https://openapi.naver.com/v1/search/news.json?query={quote(q)}&display={display}&start={start}&sort={SORT}"
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                print("WARN", q, start, r.status_code, r.text[:200])
                break
            data = r.json()
            items = data.get("items", [])
            if not items:
                break

            for it in items:
                link = it.get("link","")
                if not link:
                    continue
                k = sha1(link)
                if k in seen:
                    continue
                seen.add(k)

                title = strip_html(it.get("title",""))
                desc  = strip_html(it.get("description",""))
                pub   = it.get("pubDate","")
                # pubDate 예: "Mon, 13 Jan 2026 12:34:56 +0900"
                try:
                    published_at = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").isoformat()
                except Exception:
                    published_at = ""

                rec = {
                    "doc_id": k,
                    "source": "naver_news",
                    "source_type": "news",
                    "url": link,
                    "title": title,
                    "text": (title + "\n" + desc).strip(),
                    "lang": "ko",
                    "published_at": published_at,
                    "author": it.get("originallink",""),
                    "tags": [q],
                    "metrics": {},
                    "rights": {"allow_quote": True, "note": "원문 링크 기반 요약/인용만"},
                    "fetched_at": datetime.now().isoformat(timespec="seconds"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written >= MAX_ITEMS:
                    break

            start += display
            time.sleep(0.25)

        if written >= MAX_ITEMS:
            break

print(f"✅ wrote={written} -> {OUT}")

# rag/crawl_url.py
from __future__ import annotations

import os
import re
import json
import hashlib
import datetime
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ------------------------------------------------------------
# basic utils
# ------------------------------------------------------------
def _now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    junk = [
        "무단 전재-재배포", "AI 학습 및 활용 금지", "기사제보", "구독", "좋아요",
        "이전 다음", "기자 프로필", "언론사홈", "댓글", "공유", "추천"
    ]
    for j in junk:
        s = s.replace(j, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _detect_source(url: str) -> str:
    u = (url or "").lower()
    if "n.news.naver.com" in u or "news.naver.com" in u:
        return "naver_news"
    if "blog.naver.com" in u:
        return "naver_blog"
    if "map.kakao.com" in u or "place.map.kakao.com" in u:
        return "kakao_places"
    if "instagram.com" in u:
        return "instagram"
    return "web"

# ------------------------------------------------------------
# shop extraction (lightweight heuristic)
# ------------------------------------------------------------
_STOP_SHOP = {
    "부산", "카페", "맛집", "추천", "후기", "리뷰", "방문기", "솔직후기",
    "디저트", "베이커리", "빵집", "제과", "브런치", "핫플", "웨이팅"
}

def _shop_from_title(title: str) -> str:
    t = _clean_text(title)
    if not t:
        return ""

    # 1) "가게명 | ..." 형태
    if "|" in t:
        left = t.split("|", 1)[0].strip()
        left = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", left).strip()
        if 2 <= len(left) <= 30 and left not in _STOP_SHOP:
            return left

    # 2) "... : 가게명"
    if ":" in t:
        after = t.split(":")[-1].strip()
        after = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", after).strip()
        if 2 <= len(after) <= 30 and after not in _STOP_SHOP:
            return after

    # 3) "가게명 후기" 패턴
    m = re.search(r"([가-힣A-Za-z0-9_]{2,25})\s*(?:솔직후기|후기|리뷰|방문기)\b", t)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    return ""

def _shop_from_text(text: str) -> str:
    s = _clean_text(text)
    if not s:
        return ""

    # "카페/베이커리/제과점 XXX" 패턴
    m = re.search(r"(?:카페|베이커리|제과점)\s*([가-힣A-Za-z0-9_]{2,25})", s)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    # "XXX(카페|베이커리)" 패턴
    m = re.search(r"([가-힣A-Za-z0-9_]{2,25})\s*\((?:카페|베이커리|제과점)\)", s)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    return ""

def _extract_shop(title: str, text: str) -> str:
    shop = _shop_from_title(title)
    if shop:
        return shop
    shop = _shop_from_text(text)
    if shop:
        return shop
    return ""

# ------------------------------------------------------------
# naver news extraction
# ------------------------------------------------------------
def _extract_naver_article(soup: BeautifulSoup) -> Dict[str, str]:
    title = ""
    ogt = soup.select_one("meta[property='og:title']")
    if ogt and ogt.get("content"):
        title = ogt["content"].strip()
    if not title:
        h = soup.select_one("h2#title_area, h2.media_end_head_headline, h1")
        if h:
            title = h.get_text(" ", strip=True)

    body_node = soup.select_one("#dic_area") or soup.select_one("#newsct_article")
    text = ""
    if body_node:
        for bad in body_node.select("script, style, noscript, figure, img, iframe"):
            bad.decompose()
        text = body_node.get_text("\n", strip=True)
    else:
        article = soup.select_one("article")
        if article:
            for bad in article.select("script, style, noscript, figure, img, iframe"):
                bad.decompose()
            text = article.get_text("\n", strip=True)

    return {"title": _clean_text(title), "text": _clean_text(text)}

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def crawl_once(url: str, out_dir: str, timeout: int = 15) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    doc_id = _sha1(url)
    out_path = os.path.join(out_dir, f"{doc_id}.json")

    headers = {"User-Agent": UA}

    meta: Dict[str, Any] = {
        "id": doc_id,
        "url": url,
        "source": _detect_source(url),
        "fetched_at": _now_iso(),
        "ok": False,
        "http_status": None,
        "title": "",
        "text": "",
        "shop": "",          # ✅ NEW
        "extract_note": "",
    }

    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        meta["http_status"] = int(getattr(r, "status_code", 0) or 0)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        if meta["source"] == "naver_news":
            ex = _extract_naver_article(soup)
            meta["title"] = ex.get("title", "")
            meta["text"] = ex.get("text", "")
            meta["extract_note"] = "naver: dic_area/newsct_article"
        else:
            ogt = soup.select_one("meta[property='og:title']")
            if ogt and ogt.get("content"):
                meta["title"] = _clean_text(ogt["content"])
            # title fallback
            if not meta["title"]:
                h = soup.select_one("h1, h2, title")
                if h:
                    meta["title"] = _clean_text(h.get_text(" ", strip=True))

            body = soup.select_one("body")
            if body:
                for bad in body.select("script, style, noscript"):
                    bad.decompose()
                meta["text"] = _clean_text(body.get_text("\n", strip=True))[:20000]
            meta["extract_note"] = "generic"

        meta["ok"] = bool(meta["text"])
        meta["shop"] = _extract_shop(meta.get("title", ""), meta.get("text", ""))  # ✅ NEW

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "meta": meta}, f, ensure_ascii=False, indent=2)

        return {"ok": True, "saved": out_path, "meta": meta}

    except Exception as e:
        meta["extract_note"] = f"error: {repr(e)}"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"ok": False, "meta": meta}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return {"ok": False, "saved": out_path, "meta": meta}

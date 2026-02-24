# rag/sources/naver_news.py
from __future__ import annotations

import os
import re
import time
import argparse
from typing import List, Set, Dict, Any

import requests
from bs4 import BeautifulSoup

from rag.crawl_url import crawl_once

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 네이버 뉴스 섹션(대분류) 페이지
# 100 정치 / 101 경제 / 102 사회 / 103 생활/문화 / 104 세계 / 105 IT/과학
SECTION_URL = "https://news.naver.com/section/{sid1}"


def _norm_url(u: str) -> str:
    u = u.strip()
    if u.startswith("//"):
        u = "https:" + u
    return u


def _is_naver_article_url(u: str) -> bool:
    u = u.lower()
    return "n.news.naver.com" in u and "/article/" in u


def _extract_article_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    for a in soup.select("a[href]"):
        href = _norm_url(a.get("href", ""))
        if _is_naver_article_url(href):
            # URL 뒤 쿼리 제거(중복 줄이기)
            href = href.split("#")[0]
            # 쿼리는 남겨도 되지만 중복이 많아서 제거 추천
            href = href.split("?")[0]
            links.add(href)

    return list(links)


def fetch_section_urls(sid1: int, pages: int = 5, timeout: int = 15) -> List[str]:
    headers = {"User-Agent": UA}
    all_links: List[str] = []
    seen: Set[str] = set()

    # section/{sid1}?page= 형태가 동작하는 경우가 많음 (막히면 pages=1로라도 수집)
    for p in range(1, max(1, int(pages)) + 1):
        url = SECTION_URL.format(sid1=int(sid1))
        if p > 1:
            url = url + f"?page={p}"

        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()

        links = _extract_article_links(r.text)
        for u in links:
            if u not in seen:
                seen.add(u)
                all_links.append(u)

        # 너무 공격적이면 차단/캡차 나올 수 있음
        time.sleep(0.25)

    return all_links


def collect_naver_daily(
    out_dir: str,
    sid1_list: List[int],
    target: int = 200,
    pages_per_section: int = 6,
    sleep_sec: float = 0.35,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    # 1) 섹션에서 URL 후보 모으기
    pool: List[str] = []
    for sid1 in sid1_list:
        try:
            urls = fetch_section_urls(sid1=sid1, pages=pages_per_section)
            pool.extend(urls)
        except Exception as e:
            print(f"[WARN] section {sid1} fetch failed: {repr(e)}")

    # 중복 제거(순서 유지)
    seen: Set[str] = set()
    dedup_pool: List[str] = []
    for u in pool:
        if u not in seen:
            seen.add(u)
            dedup_pool.append(u)

    print(f"[INFO] url_pool={len(dedup_pool)} from sections={sid1_list}")

    # 2) crawl_once로 저장
    saved = 0
    failed = 0
    for u in dedup_pool:
        if saved >= int(target):
            break
        res = crawl_once(u, out_dir=out_dir)
        if res.get("ok"):
            saved += 1
        else:
            failed += 1

        if (saved + failed) % 20 == 0:
            print(f"[PROG] saved={saved} failed={failed}")

        time.sleep(max(0.05, float(sleep_sec)))

    return {"ok": True, "saved": saved, "failed": failed, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join("rag", "data"))
    ap.add_argument("--sections", default="100,101,102")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    sid1_list = [int(x.strip()) for x in str(args.sections).split(",") if x.strip().isdigit()]
    if not sid1_list:
        sid1_list = [100, 101, 102]

    res = collect_naver_daily(
        out_dir=args.out_dir,
        sid1_list=sid1_list,
        target=args.target,
        pages_per_section=args.pages,
        sleep_sec=args.sleep,
    )
    print(res)


if __name__ == "__main__":
    main()

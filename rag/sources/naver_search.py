# rag/sources/naver_search.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
from typing import Any, Dict, List, Optional

import requests

NAVER_API_BASE = "https://openapi.naver.com/v1/search"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _clean(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_html(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", "", s)
    return _clean(s)


def _headers() -> Dict[str, str]:
    cid = os.environ.get("NAVER_CLIENT_ID", "").strip()
    csec = os.environ.get("NAVER_CLIENT_SECRET", "").strip()
    return {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
    }


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ----------------------------
# ✅ (NEW) 광고/이벤트성 가벼운 컷
# ----------------------------
def _looks_like_ad(text: str) -> bool:
    t = text or ""
    if t.count("#") >= 3:
        return True
    ad_words = ["협찬", "광고", "체험단", "이벤트", "무료제공", "원고료", "파트너스"]
    return any(w in t for w in ad_words)


# ----------------------------
# ✅ (NEW) title/desc에서 shop 후보 추출(완전 정확 X, 그래도 meta에 박아두면 도움 됨)
# ----------------------------
_STOP_WORDS = {
    "부산", "카페", "맛집", "추천", "후기", "솔직후기", "리뷰", "방문기",
    "디저트", "베이커리", "빵집", "찻집", "웨이팅", "대기", "팝업", "신상",
    "두쫀쿠", "두콘쭈", "두바이", "쫀득", "쿠키",
}

def _guess_shop(title: str, desc: str) -> str:
    t = _clean(title)
    d = _clean(desc)
    txt = f"{t} {d}".strip()

    # "OOO : 가게명" 패턴
    if ":" in t:
        cand = t.split(":")[-1].strip()
        cand = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", cand).strip()
        if 2 <= len(cand) <= 25 and cand not in _STOP_WORDS:
            return cand

    # "OOO | ..." 왼쪽이 가게명인 경우
    if "|" in t:
        cand = t.split("|", 1)[0].strip()
        cand = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", cand).strip()
        if 2 <= len(cand) <= 25 and cand not in _STOP_WORDS:
            return cand

    # "(카페|베이커리|제과점) OOO" 패턴
    m = re.search(r"(?:카페|베이커리|제과점)\s*([가-힣A-Za-z0-9_]{2,20})", txt)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_WORDS:
            return cand

    # fallback: 마지막 토큰(너무 조잡하면 빈 문자열)
    tokens = re.split(r"[ \-\|\[\]\(\)<>/]+", t)
    tokens = [x.strip() for x in tokens if x.strip()]
    for tok in reversed(tokens):
        if tok in _STOP_WORDS:
            continue
        if tok in {"팝업", "후기", "추천", "웨이팅", "솔직후기", "리뷰", "방문기"}:
            continue
        if tok.endswith(("후기", "리뷰")):
            continue
        if re.fullmatch(r"[가-힣A-Za-z0-9_]{2,25}", tok):
            return tok

    return ""


def search(
    endpoint: str,  # "blog" or "local" or "cafearticle"
    query: str,
    start: int = 1,
    display: int = 10,
    sort: str = "sim",
) -> Dict[str, Any]:
    endpoint = (endpoint or "").strip().lower()
    url = f"{NAVER_API_BASE}/{endpoint}.json"
    params = {"query": query, "start": int(start), "display": int(display), "sort": sort}
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    return {"status": r.status_code, "json": (r.json() if r.ok else None), "text": r.text[:500]}


def _save_json(out_dir: str, rec: Dict[str, Any]) -> str:
    _safe_mkdir(out_dir)
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    url = str(meta.get("url", "")).strip()
    doc_id = str(rec.get("id", "")).strip() or _sha1(json.dumps(rec, ensure_ascii=False))
    fname = _sha1(url or doc_id) + ".json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return path


def run(
    out_dir: str,
    endpoint: str,
    queries: List[str],
    target: int = 200,
    display: int = 10,
    sleep: float = 0.35,
    sort: str = "sim",              # ✅ (NEW) sort 외부에서 선택 가능
    ad_filter: bool = True,         # ✅ (NEW) 광고컷 on/off
) -> Dict[str, Any]:
    cid = os.environ.get("NAVER_CLIENT_ID", "").strip()
    csec = os.environ.get("NAVER_CLIENT_SECRET", "").strip()
    if not (cid and csec):
        return {"ok": False, "error": "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET env vars not set"}

    endpoint = (endpoint or "").strip().lower()
    if endpoint not in ("blog", "local", "cafearticle"):   # ✅ (FIX) cafearticle 추가
        return {"ok": False, "error": "endpoint must be blog or local or cafearticle"}

    _safe_mkdir(out_dir)
    saved = 0
    failed = 0
    seen = set()

    try:
        for fn in os.listdir(out_dir):
            if fn.endswith(".json"):
                seen.add(fn)
    except Exception:
        pass

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue

        start = 1
        while saved < target:
            resp = search(endpoint=endpoint, query=q, start=start, display=display, sort=sort)
            if resp.get("status") != 200 or not isinstance(resp.get("json"), dict):
                failed += 1
                time.sleep(max(0.05, float(sleep)))
                break

            j = resp["json"]
            items = j.get("items", [])
            if not isinstance(items, list) or not items:
                break

            for it in items:
                if saved >= target:
                    break
                if not isinstance(it, dict):
                    continue

                link = _clean(it.get("link", "")) or _clean(it.get("originallink", ""))
                title = _strip_html(it.get("title", "")) or "(no title)"
                desc = _strip_html(it.get("description", ""))

                joined = f"{title} {desc}"
                if ad_filter and _looks_like_ad(joined):
                    continue

                shop = _guess_shop(title, desc)

                rec = {
                    "id": f"naver_{endpoint}_{_sha1(link or (title+desc))}",
                    "meta": {
                        "source": f"naver_{endpoint}",
                        "url": link,
                        "title": title,
                        "query": q,
                        "fetched_at": _now_iso(),
                        "ok": True,
                        "shop": shop,            # ✅ (NEW) meta에 shop 후보 저장
                    },
                    "shop": shop,                # ✅ (NEW) 최상위에도 shop 저장 (api.py에서 h.get('shop')로 바로 쓰려고)
                    "text": f"{title}\n- 요약: {desc}\n- 링크: {link}\n- 키워드: {q}",
                }

                fname = _sha1(link or rec["id"]) + ".json"
                if fname in seen:
                    continue

                try:
                    _save_json(out_dir, rec)
                    seen.add(fname)
                    saved += 1
                except Exception:
                    failed += 1

            if saved and saved % 20 == 0:
                print(f"[PROG] saved={saved} failed={failed} (endpoint={endpoint}, query={q}, start={start})")

            start += display
            time.sleep(max(0.05, float(sleep)))

        if saved >= target:
            break

    return {
        "ok": True,
        "saved": saved,
        "failed": failed,
        "out_dir": out_dir,
        "endpoint": endpoint,
        "queries": queries,
        "sort": sort,
        "ad_filter": ad_filter,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="blog", choices=["blog", "local", "cafearticle"])  # ✅ (FIX)
    p.add_argument("--out_dir", default=os.path.join("rag", "data", "naver"))
    p.add_argument("--queries", default="부산 두쫀쿠,부산 두콘쭈,부산 디저트 맛집,부산 카페,부산 놀거리")
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--display", type=int, default=10)
    p.add_argument("--sleep", type=float, default=0.35)
    p.add_argument("--sort", default="sim", choices=["sim", "date"])   # ✅ (NEW)
    p.add_argument("--no_ad_filter", action="store_true")              # ✅ (NEW) 옵션
    args = p.parse_args()

    queries = [s.strip() for s in str(args.queries).split(",") if s.strip()]
    result = run(
        out_dir=args.out_dir,
        endpoint=args.endpoint,
        queries=queries,
        target=args.target,
        display=args.display,
        sleep=args.sleep,
        sort=str(args.sort),
        ad_filter=(not bool(args.no_ad_filter)),
    )
    print(result)


if __name__ == "__main__":
    main()

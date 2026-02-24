# rag/sources/kakao_places.py
from __future__ import annotations

import os
import re
import json
import time
import math
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_QUERIES = [
    "부산 두쫀쿠",
    "부산 두콘쭈",
    "부산 디저트 맛집",
    "부산 카페",
    "부산 놀거리",
    "부산 전시",
    "부산 산책 코스",
    "부산 데이트 코스",
]

KAKAO_LOCAL_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _kakao_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"KakaoAK {api_key}"}


def _kakao_keyword_search(
    api_key: str,
    query: str,
    page: int = 1,
    size: int = 15,
    x: Optional[float] = None,
    y: Optional[float] = None,
    radius: Optional[int] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"query": query, "page": page, "size": size}
    if x is not None and y is not None:
        params["x"] = float(x)
        params["y"] = float(y)
    if radius is not None:
        params["radius"] = int(radius)

    r = requests.get(
        KAKAO_LOCAL_KEYWORD_URL,
        headers=_kakao_headers(api_key),
        params=params,
        timeout=30,
    )
    return {"status": r.status_code, "json": (r.json() if r.ok else None), "text": r.text[:500]}


def _doc_to_record(doc: Dict[str, Any], query: str) -> Dict[str, Any]:
    # 카카오 place 고유값: id (string)
    place_id = str(doc.get("id", "")).strip()
    place_name = _clean_text(doc.get("place_name", ""))
    category = _clean_text(doc.get("category_name", ""))
    phone = _clean_text(doc.get("phone", ""))
    address = _clean_text(doc.get("address_name", ""))
    road_address = _clean_text(doc.get("road_address_name", ""))
    place_url = _clean_text(doc.get("place_url", ""))
    x = _to_float(doc.get("x", 0.0))
    y = _to_float(doc.get("y", 0.0))
    distance = _to_int(doc.get("distance", 0))

    # RAG에 잘 먹히게 “추천 상황”을 text에 넣어줌(간단 규칙)
    # (나중에 네 감정축/액션별로 더 정교화 가능)
    recommend = []
    q = query.lower()
    if any(k in q for k in ["우울", "리프레시", "산책", "바람"]):
        recommend.append("리프레시/산책")
    if any(k in q for k in ["데이트", "기분", "설레"]):
        recommend.append("데이트/분위기")
    if any(k in q for k in ["카페", "디저트", "맛집", "두쫀쿠", "두콘쭈"]):
        recommend.append("먹거리/디저트")
    if any(k in q for k in ["전시", "놀거리"]):
        recommend.append("놀거리/체험")

    recommend_str = ", ".join(recommend) if recommend else "가벼운 외출"

    # 텍스트는 “한 번에 후보로 쓸 수 있게” 구성
    text = (
        f"{place_name} | {category}\n"
        f"- 주소: {road_address or address}\n"
        f"- 전화: {phone or '-'}\n"
        f"- 링크: {place_url or '-'}\n"
        f"- 추천상황: {recommend_str}\n"
        f"- 키워드: {query}\n"
    ).strip()

    rec = {
        "id": f"kakao_{place_id}" if place_id else f"kakao_{_sha1(place_name + place_url)}",
        "meta": {
            "source": "kakao_places",
            "url": place_url,
            "title": place_name,
            "category": category,
            "phone": phone,
            "addr": road_address or address,
            "x": x,
            "y": y,
            "distance_m": distance,
            "query": query,
            "fetched_at": _now_iso(),
            "ok": True,
        },
        "text": text,
    }
    return rec


def _save_json(out_dir: str, rec: Dict[str, Any]) -> str:
    _safe_mkdir(out_dir)

    # url이 있으면 url 기반, 없으면 id 기반으로 파일명 안정화
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
    queries: List[str],
    target: int = 200,
    pages: int = 6,
    sleep: float = 0.35,
    size: int = 15,
    x: Optional[float] = None,
    y: Optional[float] = None,
    radius: Optional[int] = None,
) -> Dict[str, Any]:
    api_key = os.environ.get("KAKAO_REST_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "KAKAO_REST_API_KEY env var not set"}

    out_dir = str(out_dir)
    _safe_mkdir(out_dir)

    saved = 0
    failed = 0
    seen_ids = set()

    # 이미 저장된 파일이 있으면 중복 방지
    try:
        for fn in os.listdir(out_dir):
            if fn.endswith(".json"):
                seen_ids.add(fn)
    except Exception:
        pass

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue

        for page in range(1, max(1, int(pages)) + 1):
            if saved >= target:
                break

            resp = _kakao_keyword_search(api_key, q, page=page, size=size, x=x, y=y, radius=radius)
            if resp.get("status") != 200 or not isinstance(resp.get("json"), dict):
                failed += 1
                time.sleep(max(0.05, float(sleep)))
                continue

            j = resp["json"]
            docs = j.get("documents", [])
            if not isinstance(docs, list) or not docs:
                # 더 이상 결과 없음
                break

            for d in docs:
                if saved >= target:
                    break
                if not isinstance(d, dict):
                    continue

                rec = _doc_to_record(d, query=q)
                # 파일명 기준 dedupe (url/id sha1)
                meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
                url = str(meta.get("url", "")).strip()
                doc_id = str(rec.get("id", "")).strip()
                fname = _sha1(url or doc_id) + ".json"
                if fname in seen_ids:
                    continue

                try:
                    _save_json(out_dir, rec)
                    seen_ids.add(fname)
                    saved += 1
                except Exception:
                    failed += 1

            if saved and saved % 20 == 0:
                print(f"[PROG] saved={saved} failed={failed} (last_query={q}, page={page})")

            time.sleep(max(0.05, float(sleep)))

        if saved >= target:
            break

    return {"ok": True, "saved": saved, "failed": failed, "out_dir": out_dir, "queries": queries}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join("rag", "data", "places"))
    p.add_argument("--queries", default=",".join(DEFAULT_QUERIES), help="comma-separated queries")
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--pages", type=int, default=6)
    p.add_argument("--sleep", type=float, default=0.35)
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--x", type=float, default=None)
    p.add_argument("--y", type=float, default=None)
    p.add_argument("--radius", type=int, default=None)
    args = p.parse_args()

    queries = [s.strip() for s in str(args.queries).split(",") if s.strip()]
    result = run(
        out_dir=args.out_dir,
        queries=queries,
        target=args.target,
        pages=args.pages,
        sleep=args.sleep,
        size=args.size,
        x=args.x,
        y=args.y,
        radius=args.radius,
    )
    print(result)


if __name__ == "__main__":
    main()

# rag/tools/build_places_jsonl.py
from __future__ import annotations

import os
import re
import json
import glob
from typing import Dict, Any, List, Tuple, Optional

VERSION = "build_places_jsonl_v2_id_fix"

DEFAULT_IN_GLOB = r"rag\data\kakao_places\*.jsonl"
DEFAULT_OUT = r"rag\data\places\places.jsonl"


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # ✅ BOM 안전
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _extract_place_id(rec: Dict[str, Any]) -> str:
    """
    ✅ 우선순위:
    1) doc_id가 숫자면 doc_id
    2) url 끝 숫자 추출
    3) chunk_id 앞부분(숫자) 추출
    4) 없으면 ""
    """
    doc_id = str(rec.get("doc_id", "") or "").strip()
    if doc_id.isdigit():
        return doc_id

    url = str(rec.get("url", "") or "").strip()
    m = re.search(r"/(\d+)$", url)
    if m:
        return m.group(1)

    chunk_id = str(rec.get("chunk_id", "") or "").strip()
    m2 = re.match(r"^(\d+)::", chunk_id)
    if m2:
        return m2.group(1)

    return ""


def _to_area(address: str, title: str, query_hint: str) -> str:
    txt = f"{address} {title} {query_hint}"

    hot = [
        "해운대", "광안리", "전포", "서면", "남포", "송정", "기장", "동래", "연산",
        "수영", "민락", "초량", "영도", "부전", "온천장", "다대포", "사상", "사하",
        "강서", "금정", "연제", "부산진", "동구", "서구", "중구", "남구", "북구"
    ]
    for k in hot:
        if k in txt:
            if k == "수영":
                return "수영구"
            if k == "부산진":
                return "부산진구"
            return k

    m = re.search(r"(부산\s*)?([가-힣]{2,4}구)", txt)
    if m:
        return m.group(2)

    return "UNKNOWN"


def _to_type(category: str, query_hint: str, text: str) -> str:
    txt = f"{category} {query_hint} {text}"

    # 세부 디저트 우선
    if "마카롱" in txt:
        return "마카롱"
    if "젤라또" in txt:
        return "젤라또"
    if "푸딩" in txt:
        return "푸딩"
    if "휘낭시에" in txt:
        return "휘낭시에"
    if "타르트" in txt:
        return "타르트"
    if "쿠키" in txt:
        return "쿠키"
    if "케이크" in txt:
        return "케이크"

    # 큰 타입
    if "제과" in txt or "베이커리" in txt or "빵" in txt:
        return "베이커리"
    if "카페" in txt or "커피" in txt:
        return "카페"
    if "디저트" in txt:
        return "디저트"

    return "카페"


def main() -> None:
    in_glob = (os.environ.get("PLACES_IN_GLOB") or DEFAULT_IN_GLOB).strip()
    out_path = (os.environ.get("PLACES_OUT") or DEFAULT_OUT).strip()

    files = sorted(glob.glob(in_glob))
    if not files:
        raise SystemExit(f"[ERR] no input files matched: {in_glob}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    raw_rows = 0
    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for fp in files:
        rows = _read_jsonl(fp)
        raw_rows += len(rows)

        for r in rows:
            if not isinstance(r, dict):
                continue

            title = _clean(str(r.get("title", "") or ""))
            text = _clean(str(r.get("text", "") or ""))
            shop = _clean(str(r.get("shop", "") or ""))
            address = _clean(str(r.get("address", "") or ""))
            category = _clean(str(r.get("category", "") or ""))
            phone = _clean(str(r.get("phone", "") or ""))
            url = _clean(str(r.get("url", "") or ""))
            x = _clean(str(r.get("x", "") or ""))
            y = _clean(str(r.get("y", "") or ""))
            query_hint = _clean(str(r.get("query", "") or ""))

            place_id = _extract_place_id(r)  # ✅ 여기서 100% 숫자 추출 시도

            # ✅ 중복 제거 키: place_id 우선, 없으면 url
            key = place_id or url or (shop + "|" + address)
            key = key.strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)

            area = _to_area(address, title, query_hint)
            typ = _to_type(category, query_hint, text)

            name = title if title else shop
            if not name:
                continue

            cleaned.append(
                {
                    "source": "kakao",
                    "id": place_id,  # ✅ 이제 빈값이면 진짜 추출 실패한 케이스뿐
                    "name": name,
                    "shop": shop,
                    "category": category,
                    "type": typ,
                    "area": area,
                    "address": address,
                    "phone": phone,
                    "url": url,
                    "x": x,
                    "y": y,
                    "query_hint": query_hint,
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stats
    def _top(items: List[str], k: int = 10) -> List[Tuple[str, int]]:
        d: Dict[str, int] = {}
        for it in items:
            d[it] = d.get(it, 0) + 1
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    areas = [r["area"] for r in cleaned]
    types = [r["type"] for r in cleaned]
    empty_ids = sum(1 for r in cleaned if not str(r.get("id", "")).strip())

    print(f"[OK] wrote: {os.path.abspath(out_path)}")
    print(f"[VER] {VERSION}")
    print(f"[STAT] input_files: {len(files)}")
    print(f"[STAT] raw_rows: {raw_rows}")
    print(f"[STAT] cleaned_rows: {len(cleaned)}")
    print(f"[STAT] empty_ids: {empty_ids}")
    print("[STAT] top areas:")
    for a, n in _top(areas, 12):
        print(f"   {a} {n}")
    print("[STAT] top types:")
    for t, n in _top(types, 12):
        print(f"   {t} {n}")


if __name__ == "__main__":
    main()

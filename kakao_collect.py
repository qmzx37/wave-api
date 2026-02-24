# kakao_collect.py (patched: gija-first + only-mode + append control)
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests

API_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

# ============================================================
# Utils
# ============================================================
def now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def getenv_key() -> str:
    """
    ✅ 너는 $env:KAKAO_REST_KEY 사용
    """
    key = (os.getenv("KAKAO_REST_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "KAKAO_REST_KEY 환경변수가 비어있음. PowerShell에서:\n"
            "$env:KAKAO_REST_KEY='YOUR_REST_KEY'\n"
        )
    return key

def kakao_headers(rest_key: str) -> Dict[str, str]:
    # 헤더 값은 반드시 ASCII여야 함. (지금처럼 env에서 키만 넣으면 OK)
    return {"Authorization": f"KakaoAK {rest_key}"}

def safe_get_str(d: Dict[str, Any], k: str) -> str:
    v = d.get(k)
    if v is None:
        return ""
    return str(v).strip()

def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_query(q: str) -> str:
    q = collapse_ws(q)
    # "카페 카페", "맛집 맛집" 같은 중복 제거 (연속 중복만)
    tokens = q.split(" ")
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return " ".join(out).strip()

def env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

# ============================================================
# ✅ Category → type 정규화 맵
# ============================================================
TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("베이커리", ["제과", "베이커리", "브레드", "빵집", "도넛", "페이스트리"]),
    ("디저트", ["디저트", "케이크", "마카롱", "타르트", "푸딩", "젤라또", "아이스크림", "브라우니", "휘낭시에", "에그타르트"]),
    ("카페", ["카페", "커피", "커피전문점", "로스터리", "스페셜티", "찻집", "티하우스", "전통찻집", "디저트카페"]),
]

# ✅ "진짜 가게" 허용 키워드(카테고리)
ALLOW_CATEGORY_KEYS = [
    "카페", "커피전문점", "로스터리", "전통찻집", "찻집",
    "제과", "베이커리", "도넛", "디저트", "아이스크림", "젤라또",
    "케이크", "마카롱", "타르트", "푸딩",
]

# ❌ 제외 카테고리 키워드(명백히 카페/디저트/베이커리 아님)
BLOCK_CATEGORY_KEYS = [
    "숙박", "호텔", "모텔", "펜션",
    "부동산", "학원",
    "병원", "약국",
    "은행", "관공서",
    "주유소", "세차", "주차장",
    "마트", "편의점",
    "미용실", "네일", "피부",
    "헬스", "요가", "필라테스",
    "PC방",
]

# ❌ “장소처럼 나오지만 가게가 아닌 것” 이름 패턴
BLOCK_NAME_PATTERNS = [
    r"\b역\b",
    r"해수욕장",
    r"\b공원\b",
    r"\b광장\b",
    r"\b시장\b",
    r"\b터미널\b",
    r"\b항\b",
]

def normalize_type(category_name: str, place_name: str, query_hint: str) -> str:
    blob = collapse_ws(f"{category_name} {place_name} {query_hint}")
    for t, keys in TYPE_RULES:
        for k in keys:
            if k in blob:
                return t
    if ("카페" in category_name) or ("커피" in category_name):
        return "카페"
    return "카페"

def is_allowed_place(place_name: str, category_name: str, address: str) -> bool:
    """
    ✅ '진짜 가게'만 남기기 위한 필터 (너무 과격하게 안 함)
    """
    name = collapse_ws(place_name)
    cat = collapse_ws(category_name)
    addr = collapse_ws(address)

    if not name:
        return False
    if len(name) < 2:
        return False

    # 주소가 부산이 아니면 제거(가끔 튐)
    if addr and ("부산" not in addr):
        return False

    # 카테고리 차단
    if any(k in cat for k in BLOCK_CATEGORY_KEYS):
        return False

    # 허용 키워드가 하나도 없으면 제거
    if not any(k in cat for k in ALLOW_CATEGORY_KEYS):
        return False

    # 이름이 너무 일반명사면 제거
    if name in {"카페", "커피", "베이커리", "디저트"}:
        return False

    # 이름 패턴 차단
    for pat in BLOCK_NAME_PATTERNS:
        if re.search(pat, name):
            return False

    return True

# ============================================================
# Kakao API
# ============================================================
def request_page(rest_key: str, query: str, page: int, size: int = 15) -> Dict[str, Any]:
    r = requests.get(
        API_URL,
        headers=kakao_headers(rest_key),
        params={"query": query, "page": page, "size": size},
        timeout=15,
    )
    try:
        js = r.json()
    except Exception:
        js = {"_raw": r.text}
    return {"status_code": r.status_code, "json": js}

def to_meta(doc: Dict[str, Any], query_hint: str) -> Dict[str, Any]:
    place_name = safe_get_str(doc, "place_name")
    place_url = safe_get_str(doc, "place_url")
    address = safe_get_str(doc, "road_address_name") or safe_get_str(doc, "address_name")
    category = safe_get_str(doc, "category_name")
    phone = safe_get_str(doc, "phone")
    x = safe_get_str(doc, "x")  # longitude
    y = safe_get_str(doc, "y")  # latitude

    kakao_id = safe_get_str(doc, "id")
    doc_id = kakao_id or sha1(place_url or (place_name + "|" + address))

    typ = normalize_type(category, place_name, query_hint)

    text = " | ".join(
        [p for p in [
            f"가게명: {place_name}" if place_name else "",
            f"카테고리: {category}" if category else "",
            f"주소: {address}" if address else "",
            f"전화: {phone}" if phone else "",
            f"좌표: {y},{x}" if (x and y) else "",
            f"쿼리: {query_hint}" if query_hint else "",
            f"type: {typ}" if typ else "",
        ] if p]
    ).strip()

    title = f"{query_hint} | {place_name}".strip(" |")

    return {
        "chunk_id": f"{doc_id}::c0",
        "doc_id": doc_id,
        "chunk_idx": 0,
        "url": place_url,
        "source": "kakao_places",
        "title": title,
        "fetched_at": now_iso(),
        "text": text,
        "shop": place_name,
        "address": address,
        "category": category,
        "phone": phone,
        "x": x,
        "y": y,
        "query": query_hint,
        "type": typ,
    }

def collect_query(
    rest_key: str,
    query: str,
    out_path: str,
    *,
    max_pages: int = 3,
    size: int = 15,
    sleep: float = 0.25,
    seen_ids: Optional[set] = None,
) -> int:
    seen_ids = seen_ids or set()
    total_written = 0

    for page in range(1, max_pages + 1):
        resp = request_page(rest_key, query=query, page=page, size=size)
        sc = int(resp.get("status_code") or 0)
        js = resp.get("json") or {}

        if sc != 200:
            print(f"[ERR] {query} page={page} status={sc} body_keys={list(js.keys())}")
            if sc in (401, 403):
                break
            time.sleep(sleep)
            continue

        docs = js.get("documents") or []
        if not docs:
            break

        is_end = bool((js.get("meta") or {}).get("is_end"))

        with open(out_path, "a", encoding="utf-8") as f:
            for d in docs:
                place_name = safe_get_str(d, "place_name")
                category = safe_get_str(d, "category_name")
                address = safe_get_str(d, "road_address_name") or safe_get_str(d, "address_name")

                if not is_allowed_place(place_name, category, address):
                    continue

                meta = to_meta(d, query_hint=query)
                doc_id = meta.get("doc_id")

                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                total_written += 1

        time.sleep(sleep)
        if is_end:
            break

    return total_written

# ============================================================
# ✅ 부산 전 지역 + 동부권(기장/일광/정관/송정) 강화 쿼리 생성
# ============================================================
BUSAN_DISTRICTS = [
    "해운대구", "수영구", "부산진구", "남구", "동래구",
    "연제구", "사하구", "서구", "중구", "영도구",
    "강서구", "금정구", "기장군", "북구", "사상구",
]

EAST_COAST_AREAS = [
    "기장", "기장군", "기장읍",
    "일광", "일광해수욕장",
    "정관", "정관읍",
    "송정", "송정해수욕장",
    "기장 카페거리", "일광 카페",
]

DESSERT_DETAIL = [
    "카페",
    "대형카페",
    "오션뷰 카페",
    "루프탑 카페",
    "베이커리",
    "빵집",
    "크로와상",
    "휘낭시에",
    "마카롱",
    "타르트",
    "푸딩",
    "젤라또",
    "아이스크림",
    "케이크",
    "치즈케이크",
    "딸기케이크",
    "쿠키",
    "두바이쫀득쿠키",
    "브라우니",
    "도넛",
    "에그타르트",
]

QUERY_TAILS = ["카페", "맛집"]

def build_queries() -> List[str]:
    """
    ✅ 핵심:
    - KAKAO_QUERY_LIMIT로 잘릴 때도 "기장/동부권"이 먼저 들어가도록 순서 조정
    - KAKAO_ONLY / KAKAO_ONLY_MODE로 부분 실행 지원

    env:
      - KAKAO_QUERY_LIMIT (default 220)
      - KAKAO_ONLY="기장군" (문자 포함 필터)
      - KAKAO_ONLY_MODE="east" | "busan" | "all" (default all)
    """
    limit = int(os.getenv("KAKAO_QUERY_LIMIT", "220"))
    only = (os.getenv("KAKAO_ONLY") or "").strip()
    only_mode = (os.getenv("KAKAO_ONLY_MODE") or "all").strip().lower()  # east|busan|all

    queries: List[str] = []

    # (B) ✅ 동부권 “동네” 쿼리를 먼저 넣는다 (limit 잘려도 기장 생존)
    if only_mode in ("east", "all"):
        for area in EAST_COAST_AREAS:
            for d in DESSERT_DETAIL:
                for tail in QUERY_TAILS:
                    q = normalize_query(f"부산 {area} {d} {tail}")
                    queries.append(q)

    # (A) 부산 구/군 기반 쿼리
    if only_mode in ("busan", "all"):
        for gu in BUSAN_DISTRICTS:
            for d in DESSERT_DETAIL:
                for tail in QUERY_TAILS:
                    q = normalize_query(f"부산 {gu} {d} {tail}")
                    queries.append(q)

    # 중복 제거 (순서 유지)
    uniq: List[str] = []
    seen = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)

    # ✅ KAKAO_ONLY 필터 (ex: "기장군", "기장", "일광", "정관", "송정")
    if only:
        uniq = [q for q in uniq if only in q]

    # limit 적용
    return uniq[:max(1, limit)]

# ============================================================
# Main
# ============================================================
def main() -> None:
    rest_key = getenv_key()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "rag", "data", "kakao_places")
    ensure_dir(out_dir)

    # ✅ 파일명 커스터마이즈 가능 (기장만 따로 뺄 때)
    out_name = (os.getenv("KAKAO_OUT_NAME") or "").strip()
    if not out_name:
        out_name = f"kakao_places_{datetime.date.today().isoformat()}.jsonl"
    out_path = os.path.join(out_dir, out_name)

    # ✅ 기본: 같은 날짜 파일은 새로 만들기
    #    - KAKAO_APPEND=1이면 append
    if os.path.exists(out_path) and (not env_bool("KAKAO_APPEND", "0")):
        os.remove(out_path)

    queries = build_queries()
    seen_ids: set = set()

    max_pages = int(os.getenv("KAKAO_MAX_PAGES", "3"))
    size = int(os.getenv("KAKAO_PAGE_SIZE", "15"))
    sleep = float(os.getenv("KAKAO_SLEEP", "0.25"))

    print(f"[KAKAO] out_path = {os.path.abspath(out_path)}")
    print(f"[KAKAO] queries  = {len(queries)}  (limit via KAKAO_QUERY_LIMIT, only={os.getenv('KAKAO_ONLY','')}, mode={os.getenv('KAKAO_ONLY_MODE','all')})")
    print(f"[KAKAO] max_pages= {max_pages} size={size} sleep={sleep}")
    print(f"[KAKAO] start    = {now_iso()}")

    total = 0
    for q in queries:
        n = collect_query(
            rest_key=rest_key,
            query=q,
            out_path=out_path,
            max_pages=max_pages,
            size=size,
            sleep=sleep,
            seen_ids=seen_ids,
        )
        total += n
        print(f"[KAKAO] +{n:4d}  ({q})  total={total}")

    print(f"[KAKAO] done. total_written={total}  end={now_iso()}")

if __name__ == "__main__":
    main()

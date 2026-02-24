# tools/dedup_kakao_places.py
from __future__ import annotations

import os
import json
import re
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------
# 부산 구/군 정규화
# -----------------------
BUSAN_GUGUN = [
    "중구", "서구", "동구", "영도구", "부산진구", "동래구", "남구", "북구",
    "해운대구", "사하구", "금정구", "강서구", "연제구", "수영구", "사상구", "기장군",
]

# 별칭/오타/동네명 → 해당 구/군으로 매핑(필요하면 더 추가)
AREA_ALIASES: List[Tuple[str, List[str]]] = [
    ("해운대구", ["해운대", "센텀", "센텀시티", "우동", "중동", "장산", "재송", "송정(해운대)"]),
    ("수영구", ["광안리", "광안", "민락", "수영", "남천", "금련산"]),
    ("부산진구", ["서면", "전포", "전포동", "부전", "범천", "양정"]),
    ("동래구", ["동래", "온천장", "사직", "명륜"]),
    ("남구", ["대연", "용호", "문현", "경성대", "부경대"]),
    ("중구", ["남포", "남포동", "광복", "자갈치"]),
    ("영도구", ["영도", "태종대"]),
    ("동구", ["초량", "부산역", "범일"]),
    ("사하구", ["하단", "다대포", "감천"]),
    ("강서구", ["명지", "대저"]),
    ("금정구", ["장전", "부산대", "구서"]),
    ("연제구", ["연산", "연산동"]),
    ("사상구", ["사상", "괘법", "덕포"]),
    ("서구", ["송도", "암남", "송도해수욕장"]),
    ("북구", ["화명", "덕천", "구포"]),
    ("기장군", ["기장", "정관", "일광"]),
]

# -----------------------
# 타입 정규화(카테고리명/쿼리 기반)
# -----------------------
TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("베이커리", ["베이커리", "제과", "빵집", "브레드", "빵"]),
    ("디저트", ["디저트", "케이크", "타르트", "마카롱", "휘낭시에", "쿠키", "푸딩", "젤라또", "아이스크림"]),
    ("대형카페", ["대형", "루프탑", "오션뷰", "바다뷰", "뷰", "정원", "테라스"]),
    ("카페", ["카페", "커피", "티", "찻집"]),
]

# 카카오 category_name 예: "음식점 > 카페 > 커피전문점"
def normalize_type(category_name: str = "", query: str = "", place_name: str = "") -> str:
    hay = " ".join([(category_name or ""), (query or ""), (place_name or "")]).strip()
    if not hay:
        return "기타"

    # 우선순위: 베이커리 > 디저트 > 대형카페 > 카페
    for t, keys in TYPE_RULES:
        for k in keys:
            if k in hay:
                return t

    # 카카오 카테고리에 '카페'가 있으면 기본은 카페
    if "카페" in (category_name or ""):
        return "카페"
    return "기타"


def normalize_area(address: str = "", road_address: str = "", place_name: str = "", query: str = "") -> str:
    hay = " ".join([(address or ""), (road_address or ""), (place_name or ""), (query or "")]).strip()
    if not hay:
        return ""

    # 1) 정규식으로 "부산 ... (xx구/군)" 추출
    m = re.search(r"(부산|부산광역시)\s*([가-힣]+(?:구|군))", hay)
    if m:
        gugun = m.group(2)
        if gugun in BUSAN_GUGUN:
            return gugun

    # 2) 별칭 매핑
    for gugun, aliases in AREA_ALIASES:
        for a in aliases:
            if a and a in hay:
                return gugun

    # 3) "xx구" 직접 탐지 (부산이라는 단어 없이도)
    m2 = re.search(r"([가-힣]+(?:구|군))", hay)
    if m2:
        gugun = m2.group(1)
        if gugun in BUSAN_GUGUN:
            return gugun

    return ""


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_best(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    같은 place_id 중 무엇을 남길지 결정.
    - road_address_name(도로명) 있는 쪽 선호
    - phone 있는 쪽 선호
    - category_name 긴 쪽 선호
    - 그 외에는 기존 유지
    """
    def score(x: Dict[str, Any]) -> int:
        s = 0
        if (x.get("road_address_name") or x.get("road_address") or ""): s += 3
        if (x.get("phone") or ""): s += 2
        if len(str(x.get("category_name") or "")) >= 8: s += 1
        return s

    return new if score(new) > score(existing) else existing


def extract_place_id(rec: Dict[str, Any]) -> str:
    # 카카오 응답 필드는 보통 "id"
    pid = rec.get("id") or rec.get("place_id") or rec.get("kakao_id")
    if pid is None:
        return ""
    return str(pid).strip()


def to_place_doc(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    rag에서 쓰기 좋은 최소 스키마로 정리.
    """
    pid = extract_place_id(rec)

    place_name = (rec.get("place_name") or rec.get("name") or "").strip()
    address = (rec.get("address_name") or rec.get("address") or "").strip()
    road = (rec.get("road_address_name") or rec.get("road_address") or "").strip()
    phone = (rec.get("phone") or "").strip()
    category = (rec.get("category_name") or rec.get("category") or "").strip()
    place_url = (rec.get("place_url") or rec.get("url") or "").strip()
    x = rec.get("x")
    y = rec.get("y")

    # kakao_collect.py가 query를 저장해뒀다면 활용
    query = (rec.get("query") or rec.get("_query") or rec.get("q") or "").strip()

    area = normalize_area(address=address, road_address=road, place_name=place_name, query=query)
    ptype = normalize_type(category_name=category, query=query, place_name=place_name)

    # RAG chunker가 읽기 쉬운 텍스트(= text) 생성
    lines = []
    if place_name: lines.append(f"가게: {place_name}")
    if ptype: lines.append(f"종류: {ptype}")
    if area: lines.append(f"지역: 부산 {area}")
    if road: lines.append(f"주소(도로명): {road}")
    elif address: lines.append(f"주소: {address}")
    if phone: lines.append(f"전화: {phone}")
    if category: lines.append(f"카테고리: {category}")
    if place_url: lines.append(f"링크: {place_url}")

    text = "\n".join(lines).strip()

    return {
        "id": pid,
        "source": "kakao_places",
        "title": place_name or pid,
        "shop": place_name,
        "area": area,          # 예: 해운대구
        "type": ptype,         # 예: 디저트/베이커리/대형카페/카페
        "address": address,
        "road_address": road,
        "phone": phone,
        "category_name": category,
        "x": float(x) if x is not None and str(x) != "" else None,
        "y": float(y) if y is not None and str(y) != "" else None,
        "url": place_url,
        "query": query,
        "text": text,          # ✅ RAG에서 실제로 검색될 본문
        "raw": rec,            # 원본 보관(디버그용)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="kakao_collect.py 결과 jsonl 경로")
    ap.add_argument("--out", dest="out_path", default=os.path.join("rag", "data", "places", "places.jsonl"))
    ap.add_argument("--min_text", type=int, default=1, help="text 길이가 너무 짧으면 제외(기본 1)")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"input not found: {in_path}")

    # 1) id 기반 dedup
    by_id: Dict[str, Dict[str, Any]] = {}
    total = 0
    no_id = 0

    for rec in read_jsonl(in_path):
        total += 1
        pid = extract_place_id(rec)
        if not pid:
            no_id += 1
            continue
        if pid in by_id:
            by_id[pid] = pick_best(by_id[pid], rec)
        else:
            by_id[pid] = rec

    # 2) 정규화 + text 생성
    docs: List[Dict[str, Any]] = []
    area_count: Dict[str, int] = {}
    type_count: Dict[str, int] = {}

    for pid, rec in by_id.items():
        d = to_place_doc(rec)
        if len(d.get("text") or "") < int(args.min_text):
            continue
        docs.append(d)

        a = d.get("area") or ""
        t = d.get("type") or "기타"
        area_count[a] = area_count.get(a, 0) + 1
        type_count[t] = type_count.get(t, 0) + 1

    # 3) 저장
    write_jsonl(out_path, docs)

    # 4) 요약 출력
    print("[DONE]")
    print(f"input_total     = {total}")
    print(f"no_id_skipped   = {no_id}")
    print(f"unique_place_id = {len(by_id)}")
    print(f"written_docs    = {len(docs)}")
    print(f"out_path        = {os.path.abspath(out_path)}")
    print("\n[AREA TOP]")
    for k, v in sorted(area_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        kk = k if k else "(unknown)"
        print(f"- {kk}: {v}")
    print("\n[TYPE TOP]")
    for k, v in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

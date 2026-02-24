# naver_collect_evidence.py
# ------------------------------------------------------------
# 목적:
#   rag/data/places/places.jsonl(카카오 정제 결과)을 순회하면서
#   place마다 네이버 카페글 1개(근거)만 뽑아 jsonl로 저장한다.
#
# 핵심 목표(제품용):
#   - "후기처럼 말하는 문장형 스니펫"만 남기기
#   - 공지/양식/리스트/저정보/광고성 스니펫 강력 컷
#   - 후보 10개 중 "게이트 통과 + 점수 최고" 1개만 저장
#   - URL/텍스트 중복 방지로 인덱스 노이즈 최소화
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import time
import html
import hashlib
import datetime
from typing import Dict, Any, Iterable, Optional, Tuple, List

import requests


# ============================================================
# ENV
# ============================================================
NAVER_CLIENT_ID = (os.getenv("NAVER_CLIENT_ID") or "").strip()
NAVER_CLIENT_SECRET = (os.getenv("NAVER_CLIENT_SECRET") or "").strip()

PLACES_JSONL = os.getenv("PLACES_JSONL", r"rag\data\places\places.jsonl")
OUT_DIR = os.getenv("NAVER_EVIDENCE_DIR", r"rag\data\naver_evidence")

TODAY = datetime.date.today().isoformat()
OUT_JSONL = os.path.join(OUT_DIR, f"naver_evidence_{TODAY}.jsonl")
EVIDENCE_INDEX_JSON = os.path.join(OUT_DIR, f"evidence_index_{TODAY}.json")

NAVER_DISPLAY = int(os.getenv("NAVER_DISPLAY", "10"))
NAVER_SORT = os.getenv("NAVER_SORT", "sim")  # sim | date
SLEEP_SEC = float(os.getenv("NAVER_SLEEP", "0.25"))

LIMIT = int(os.getenv("NAVER_LIMIT", "0"))  # 0이면 전체
DEBUG = os.getenv("NAVER_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y", "on")

# ============================================================
# ✅ "좋은 근거만 저장" 게이트(제품용 고정)
# ============================================================
EVI_GOOD_ONLY = os.getenv("EVI_GOOD_ONLY", "1").strip().lower() in ("1", "true", "yes", "y", "on")

EVI_MIN_CHARS = int(os.getenv("EVI_MIN_CHARS", "80"))            # 문장형 후기 최소 길이(스니펫 기준)
EVI_MIN_KEYWORD_HITS = int(os.getenv("EVI_MIN_KW_HITS", "2"))    # (place/구/동 등) 단서 매칭 최소
EVI_MIN_SCORE = float(os.getenv("EVI_MIN_SCORE", "0.70"))        # 0~1, 문장형 후기 선호로 기본 상향
EVI_HASHTAG_CUT = int(os.getenv("EVI_HASHTAG_CUT", "8"))         # # 8개 이상 컷(완화)

EVI_ALLOW_DOMAINS = [d.strip() for d in (os.getenv("EVI_ALLOW_DOMAINS", "cafe.naver.com,blog.naver.com").split(",")) if d.strip()]
EVI_BAN_KEYWORDS = [w.strip() for w in (os.getenv(
    "EVI_BAN_KEYWORDS",
    "협찬,광고,체험단,이벤트,무료제공,원고료,파트너스,제공받아,지원받아,소정의"
).split(",")) if w.strip()]

# ✅ 템플릿/공지/양식 컷
EVI_TEMPLATE_MIN_CHARS = int(os.getenv("EVI_TEMPLATE_MIN_CHARS", "110"))  # 더 빡세게(기본 상향)
EVI_HASHTAG_SOFT_CUT = int(os.getenv("EVI_HASHTAG_SOFT_CUT", "4"))        # 짧은 글 + #많으면 컷

EVI_TEMPLATE_PHRASES = [p.strip() for p in (os.getenv(
    "EVI_TEMPLATE_PHRASES",
    "게시물 제목을 그대로 붙여넣기,글양식,양식,양식대로,작성해주세요,필수,지도(필수),하단... 쩜3개,쩜3개,"
    "방문(이용)날짜,방문(이용) 날짜,운영진,공지,공지사항,필독,후기작성,후기 작성,복사,붙여넣기,삭제하지,삭제 하지,"
    "금지매장,금지 매장,서식,체크리스트,질문,답변"
).split(",")) if p.strip()]

EVI_LOWINFO_LIST_WORDS = [p.strip() for p in (os.getenv(
    "EVI_LOWINFO_LIST_WORDS",
    "모음,추천 리스트,리스트,지도,총정리,베스트,TOP,top,정리,후기모음,모아보기"
).split(",")) if p.strip()]

# ✅ NEW: "후기처럼 말하는 문장형" 필터(핵심)
# - 이런 단어가 있어야 "체감 후기" 확률이 높음
EVI_EXPERIENCE_WORDS = [p.strip() for p in (os.getenv(
    "EVI_EXPERIENCE_WORDS",
    # 뷰/분위기/좌석/혼공/디저트 등 체감 단서 중심
    "오션뷰,바다뷰,뷰가,뷰맛집,통창,바다,해변,광안리,해운대,"
    "분위기,인테리어,조용,편안,아늑,감성,뷰좋,자리,좌석,콘센트,와이파이,"
    "혼자,혼카페,혼공,공부,작업,노트북,"
    "케이크,디저트,빵,베이커리,커피,라떼,맛집,존맛,추천,"
    "주차,웨이팅,대기,예약"
).split(",")) if p.strip()]

EVI_MIN_EXPERIENCE_HITS = int(os.getenv("EVI_MIN_EXPERIENCE_HITS", "1"))  # 체감 단서 최소 1개


# ============================================================
# NAVER API
# ============================================================
NAVER_CAFE_URL = "https://openapi.naver.com/v1/search/cafearticle.json"


def _headers() -> Dict[str, str]:
    return {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }


def _strip_tags(s: str) -> str:
    s = re.sub(r"</?b>", "", s or "", flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# ✅ must_include 강한 비교(공백/특수문자 흔들림 방지)
# ============================================================
def _norm_comp(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w가-힣]", "", s)
    return s


def _loose_include(joined: str, must_include: str) -> bool:
    mi = (must_include or "").strip()
    if not mi:
        return True

    jn = _norm_comp(joined)
    mn = _norm_comp(mi)

    if mn and (mn in jn):
        return True

    core = re.split(r"\s+|\(|\)|,|/|-", mi)[0].strip()
    cn = _norm_comp(core)
    if cn and (cn in jn):
        return True

    jn2 = _norm(joined)
    mn2 = _norm(mi)
    if mn2 and (mn2 in jn2):
        return True

    return False


# ============================================================
# ✅ 광고/협찬 감지(완화)
# ============================================================
def _looks_like_ad(text: str) -> bool:
    t = text or ""
    if t.count("#") >= EVI_HASHTAG_CUT:
        return True
    return any(w in t for w in EVI_BAN_KEYWORDS)


# ============================================================
# ✅ 템플릿/공지/양식/저정보 스니펫 컷
# ============================================================
def _looks_like_template_or_notice(text: str) -> bool:
    t = _strip_tags(text or "").strip()
    if not t:
        return True

    if any(p in t for p in EVI_TEMPLATE_PHRASES):
        return True

    # 짧은 글 + 해시태그 많음 → 홍보/템플릿 확률 높음
    if t.count("#") >= EVI_HASHTAG_SOFT_CUT and len(t) < 200:
        return True

    # 스니펫이 너무 짧으면 공지/제목 조각만 있는 경우가 많음
    if len(t) < EVI_TEMPLATE_MIN_CHARS:
        return True

    return False


def _is_low_info_snippet(title: str, desc: str) -> bool:
    txt = _strip_tags(f"{title} {desc}")
    if any(w in txt for w in EVI_LOWINFO_LIST_WORDS):
        # “단일 가게 근거”로 쓸 만한 구체 단서 없으면 리스트/모음 확률↑
        anchors = ("주소", "메뉴", "주차", "웨이팅", "대기", "좌석", "콘센트", "영업", "시간", "가격", "추천")
        if not any(a in txt for a in anchors):
            return True
    return False


# ============================================================
# ✅ NEW: "후기처럼 말하는 문장형" 스니펫 감지
# - 목표: "오션뷰라서 이쁘네요 / 케이크 맛집 / 혼자 공부 가능" 같은 문장
# ============================================================
_SENT_ENDERS = ("다.", "요.", "네요", "어요", "아요", "입니다", "했어요", "했네요", "추천", "좋아요")


def _korean_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    ko = len(re.findall(r"[가-힣]", s))
    return ko / max(1, total)


def _looks_like_reviewish_sentence(text: str) -> bool:
    t = _strip_tags(text or "").strip()
    if not t:
        return False

    # 링크/아이디/전화번호/과도한 숫자 → 후기 스니펫보단 안내문/리스트일 가능성
    if re.search(r"(http|www\.)", t, flags=re.IGNORECASE):
        return False
    if re.search(r"\b\d{2,4}-\d{3,4}-\d{4}\b", t):
        return False

    # 한국어 비율 낮으면(영문/기호 위주) 컷
    if _korean_ratio(t) < 0.35:
        return False

    # 너무 리스트처럼 생긴 형태(구분자 과다)
    if t.count("|") >= 3 or t.count("•") >= 2 or t.count("·") >= 6:
        return False

    # ✅ “체감 단서(경험 단어)” 먼저 계산 (종결어미 예외 통과에 사용)
    exp_hits = 0
    tl = t.lower()
    for w in EVI_EXPERIENCE_WORDS:
        w2 = (w or "").strip().lower()
        if w2 and w2 in tl:
            exp_hits += 1

    # ✅ 문장 종결감이 없어도 "경험단서(exp_hits)"가 충분하면 통과
    if not any(e in t for e in _SENT_ENDERS) and (t.count(".") < 1 and t.count("!") < 1 and t.count("~") < 1):
        if exp_hits < 2:
            return False

    # ✅ 경험 단서 최소 조건은 그대로 유지 (제품용 품질)
    if exp_hits < EVI_MIN_EXPERIENCE_HITS:
        return False

    return True


# ============================================================
# ✅ 좋은 근거 스코어링/게이트 (스니펫 기반)
# ============================================================
def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def keyword_hits(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    hits = 0
    for k in keywords:
        k2 = (k or "").strip().lower()
        if not k2:
            continue
        if k2 in t:
            hits += 1
    return hits


def allowed_domain(url: str) -> bool:
    if not EVI_ALLOW_DOMAINS:
        return True
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        return any(d in host for d in EVI_ALLOW_DOMAINS)
    except Exception:
        return False


def experience_hits(text: str) -> int:
    t = (text or "").lower()
    hits = 0
    for w in EVI_EXPERIENCE_WORDS:
        w2 = (w or "").strip().lower()
        if w2 and w2 in t:
            hits += 1
    return hits


def evidence_score(joined: str, query_keywords: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    0~1 점수(스니펫 기반)
    - 길이 + (place/구/동) 단서 + 경험 단서 + 광고/리스트 패널티
    """
    j = normalize_text(joined)
    n = len(j)

    # 1) 길이 점수 (스니펫이므로 240자 근처에서 포화)
    len_score = min(1.0, n / 240.0)

    # 2) 키워드 점수 (place/구/동 중심)
    hit = keyword_hits(j, query_keywords)
    kw_score = min(1.0, hit / 4.0)

    # 3) 경험 단서 보너스 (뷰/분위기/케이크/공부 등)
    exp = experience_hits(j)
    exp_score = min(1.0, exp / 3.0)  # 3개 이상이면 포화

    # 4) 광고/협찬 패널티
    ban_hits = keyword_hits(j, EVI_BAN_KEYWORDS)
    ad_penalty = min(0.65, 0.30 * ban_hits)

    # 5) 해시태그 패널티
    hashtag_cnt = j.count("#")
    tag_penalty = 0.18 if hashtag_cnt >= 6 else (0.08 if hashtag_cnt >= 4 else 0.0)

    # 6) 리스트/정리 느낌 패널티(구분자 과다)
    list_penalty = 0.12 if (j.count("|") >= 3 or j.count("•") >= 2 or j.count("·") >= 6) else 0.0

    # ✅ 최종 점수: "후기처럼 말하는 경험 단서" 비중을 가장 크게
    score = 0.30 * len_score + 0.35 * kw_score + 0.35 * exp_score
    score = max(0.0, min(1.0, score - ad_penalty - tag_penalty - list_penalty))

    feats = {
        "len": n,
        "len_score": len_score,
        "kw_hits": hit,
        "kw_score": kw_score,
        "exp_hits": exp,
        "exp_score": exp_score,
        "ban_hits": ban_hits,
        "ad_penalty": ad_penalty,
        "hashtag_cnt": hashtag_cnt,
        "tag_penalty": tag_penalty,
        "list_penalty": list_penalty,
    }
    return score, feats


def passes_gate(url: str, joined: str, score: float, feats: Dict[str, Any]) -> Tuple[bool, str]:
    if not allowed_domain(url):
        return False, "domain_block"

    if len(joined or "") < EVI_MIN_CHARS:
        return False, "too_short"

    # ✅ 키워드 히트는 "가게/지역 단서" 중심으로 최소 확보
    if feats.get("kw_hits", 0) < EVI_MIN_KEYWORD_HITS:
        return False, "low_keyword_hits"

    # ✅ 경험 단서(뷰/분위기/케이크/공부 등) 최소 확보
    if feats.get("exp_hits", 0) < EVI_MIN_EXPERIENCE_HITS:
        return False, "low_experience_hits"

    if score < EVI_MIN_SCORE:
        return False, "low_score"

    # ✅ 광고 단어는 1개만 있어도 컷(제품용: 인덱스 오염 방지)
    if feats.get("ban_hits", 0) >= 1:
        return False, "ad_suspect"

    return True, "ok"


# ============================================================
# NAVER SEARCH: 후보들 중 "게이트 통과 + 점수 최고" 1개 선택
# ============================================================
def naver_cafe_search_best(
    query: str,
    must_include: Optional[str] = None,
    query_keywords: Optional[List[str]] = None,
    seen_url: Optional[set] = None,
    seen_text_hash: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return None
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return None

    params = {
        "query": q,
        "display": max(1, min(NAVER_DISPLAY, 100)),
        "start": 1,
        "sort": NAVER_SORT,
    }

    try:
        r = requests.get(NAVER_CAFE_URL, headers=_headers(), params=params, timeout=10)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    items = data.get("items") or []
    if not items:
        return None

    mi = (must_include or "").strip()
    qk = query_keywords or []

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for it in items:
        title = _strip_tags(str(it.get("title", "")))
        desc = _strip_tags(str(it.get("description", "")))
        link = str(it.get("link", "") or "").strip()

        joined = f"{title} {desc}".strip()
        if not joined:
            continue

        # 0) URL/텍스트 중복 방지(전역)
        if seen_url is not None and link and link in seen_url:
            continue
        th = hash_text(normalize_text(joined))
        if seen_text_hash is not None and th in seen_text_hash:
            continue

        # 1) 광고컷
        if _looks_like_ad(joined):
            continue

        # 2) 공지/양식/템플릿/저정보 컷
        if _looks_like_template_or_notice(joined) or _is_low_info_snippet(title, desc):
            continue

        # 3) ✅ "후기처럼 말하는 문장형" 강제(네가 원하는 톤)
        if not _looks_like_reviewish_sentence(joined):
            continue

        # 4) must_include(완화)
        if mi and (not _loose_include(joined, mi)):
            continue

        # 5) 스코어링 + 게이트
        score, feats = evidence_score(joined, qk)
        ok, reason = passes_gate(link, joined, score, feats)

        if not ok:
            continue

        if score > best_score:
            best_score = score
            best = {
                "title": title,
                "description": desc,
                "link": link,
                "evi_score": score,
                "evi_feats": feats,
                "evi_gate_reason": reason,
            }

    return best


# ============================================================
# PLACES parsing
# ============================================================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def pick_place_key(rec: Dict[str, Any]) -> str:
    pid = str(rec.get("id") or rec.get("place_id") or rec.get("doc_id") or "").strip()
    if pid:
        return pid
    shop = str(rec.get("place_name") or rec.get("shop") or rec.get("name") or "").strip()
    addr = str(rec.get("road_address_name") or rec.get("address") or rec.get("address_name") or "").strip()
    return sha1(f"{shop}|{addr}")


# ============================================================
# 지역(구/동) 추출
# ============================================================
_GU_RE = re.compile(r"([가-힣]{1,4}구)")
_DONG_RE = re.compile(r"([가-힣]{1,4}동)")
_GUN_RE = re.compile(r"([가-힣]{1,4}군)")


def extract_gu_dong(addr: str) -> Tuple[str, str, str]:
    a = (addr or "").strip()
    gu = ""
    dong = ""
    m = _GU_RE.search(a)
    if m:
        gu = m.group(1)
    else:
        m2 = _GUN_RE.search(a)
        if m2:
            gu = m2.group(1)

    m3 = _DONG_RE.search(a)
    if m3:
        dong = m3.group(1)

    return gu, dong, a


def build_naver_query(place_name: str, addr: str) -> str:
    place_name = (place_name or "").strip()
    gu, dong, _ = extract_gu_dong(addr)

    parts: List[str] = []
    if place_name:
        parts.append(place_name)
    if gu:
        parts.append(gu)
    if dong:
        parts.append(dong)

    parts.append("후기")

    uniq: List[str] = []
    for p in parts:
        if p and p not in uniq:
            uniq.append(p)

    return " ".join(uniq).strip()


def build_query_keywords(place_name: str, addr: str) -> List[str]:
    """
    ✅ 스코어링 단서 키워드
    - place_name/브랜드코어/구/동 중심
    - "후기" 같은 과매칭 단어는 점수용에서 제거(뻥튀기 방지)
    """
    gu, dong, _ = extract_gu_dong(addr)
    kws: List[str] = []

    if place_name:
        kws.append(place_name)
        core = re.split(r"\s+|\(|\)|,|/|-", place_name)[0].strip()
        if core and core != place_name:
            kws.append(core)

    if gu:
        kws.append(gu)
    if dong:
        kws.append(dong)

    # "주차/웨이팅" 정도만 남김(후기는 제거)
    kws.extend(["주차", "웨이팅", "대기"])

    uniq: List[str] = []
    for k in kws:
        k = (k or "").strip()
        if k and k not in uniq:
            uniq.append(k)
    return uniq


# ============================================================
# OUTPUT record
# ============================================================
def to_evidence_record(place: Dict[str, Any], hit: Dict[str, Any]) -> Dict[str, Any]:
    place_id = pick_place_key(place)
    place_name = str(place.get("place_name") or place.get("shop") or place.get("name") or "").strip()
    addr = str(place.get("road_address_name") or place.get("address") or place.get("address_name") or "").strip()

    title = hit.get("title", "")
    desc = hit.get("description", "")
    link = hit.get("link", "")

    snippet = (desc or title or "").strip()
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 240:
        snippet = snippet[:240].rstrip() + "…"

    gu, dong, _ = extract_gu_dong(addr)

    rec = {
        "chunk_id": f"naver::{place_id}::c0",
        "doc_id": f"naver::{place_id}",
        "chunk_idx": 0,
        "source": "naver_cafe",
        "url": link,
        "fetched_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "title": f"{place_name} | {title}".strip(" |"),
        "text": " | ".join([p for p in [
            f"가게명: {place_name}" if place_name else "",
            f"주소: {addr}" if addr else "",
            f"지역: {gu} {dong}".strip() if (gu or dong) else "",
            f"근거요약: {snippet}" if snippet else "",
        ] if p]),
        "shop": place_name,
        "address": addr,
        "area_gu": gu,
        "area_dong": dong,
        "evidence": {
            "title": title,
            "description": desc,
            "snippet": snippet,
            "link": link,
        },
        "evi_score": float(hit.get("evi_score", 0.0)),
        "evi_feats": hit.get("evi_feats", {}),
        "evi_gate_reason": hit.get("evi_gate_reason", ""),
    }
    return rec


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not os.path.exists(PLACES_JSONL):
        raise RuntimeError(f"places.jsonl not found: {PLACES_JSONL}")

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise RuntimeError(
            "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 비어있음.\n"
            "PowerShell 예:\n"
            "$env:NAVER_CLIENT_ID='xxx'\n"
            "$env:NAVER_CLIENT_SECRET='yyy'\n"
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_JSONL) and (os.getenv("NAVER_APPEND", "0").strip().lower() not in ("1", "true", "yes", "y", "on")):
        os.remove(OUT_JSONL)

    seen_place_keys = set()
    evidence_keys = set()

    seen_urls = set()
    seen_text_hash = set()

    kept = 0
    miss = 0

    print("[NAVER] places =", os.path.abspath(PLACES_JSONL))
    print("[NAVER] out    =", os.path.abspath(OUT_JSONL))
    print("[NAVER] index  =", os.path.abspath(EVIDENCE_INDEX_JSON))
    print("[NAVER] display=", NAVER_DISPLAY, "sort=", NAVER_SORT, "sleep=", SLEEP_SEC)
    print("[EVI]  good_only=", EVI_GOOD_ONLY,
          "min_chars=", EVI_MIN_CHARS,
          "min_kw_hits=", EVI_MIN_KEYWORD_HITS,
          "min_score=", EVI_MIN_SCORE,
          "min_exp_hits=", EVI_MIN_EXPERIENCE_HITS)
    print("[EVI]  template_min_chars=", EVI_TEMPLATE_MIN_CHARS,
          "template_phrases=", len(EVI_TEMPLATE_PHRASES),
          "lowinfo_words=", len(EVI_LOWINFO_LIST_WORDS),
          "exp_words=", len(EVI_EXPERIENCE_WORDS))
    if LIMIT > 0:
        print("[NAVER] LIMIT  =", LIMIT, "(processed limit)")
    if DEBUG:
        print("[NAVER] DEBUG  = 1")

    with open(OUT_JSONL, "a", encoding="utf-8") as w:
        for i, place in enumerate(iter_jsonl(PLACES_JSONL), start=1):
            if LIMIT > 0 and i > LIMIT:
                break

            place_key = pick_place_key(place)
            if place_key in seen_place_keys:
                continue
            seen_place_keys.add(place_key)

            place_name = str(place.get("place_name") or place.get("shop") or place.get("name") or "").strip()
            addr = str(place.get("road_address_name") or place.get("address") or place.get("address_name") or "").strip()
            if not place_name:
                continue

            q = build_naver_query(place_name, addr)
            qk = build_query_keywords(place_name, addr)

            hit = naver_cafe_search_best(
                q,
                must_include=place_name,
                query_keywords=qk,
                seen_url=seen_urls,
                seen_text_hash=seen_text_hash,
            )

            if hit is None:
                miss += 1
                if DEBUG and i <= 30:
                    print(f"[DEBUG] MISS place='{place_name}' q='{q}' addr='{addr[:40]}'")
                time.sleep(SLEEP_SEC)
                continue

            link = (hit.get("link") or "").strip()
            if link:
                seen_urls.add(link)
            joined = f"{hit.get('title','')} {hit.get('description','')}".strip()
            seen_text_hash.add(hash_text(normalize_text(joined)))

            rec = to_evidence_record(place, hit)
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            evidence_keys.add(place_key)

            if (kept % 50) == 0:
                print(f"[NAVER] kept={kept} / processed={i} (last={place_name})")

            time.sleep(SLEEP_SEC)

    idx_obj = {
        "created_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "places_jsonl": os.path.abspath(PLACES_JSONL),
        "evidence_jsonl": os.path.abspath(OUT_JSONL),
        "num_places_seen": len(seen_place_keys),
        "num_evidence": kept,
        "miss": miss,
        "evidence_place_keys": sorted(list(evidence_keys)),
        "gate": {
            "good_only": EVI_GOOD_ONLY,
            "min_chars": EVI_MIN_CHARS,
            "min_keyword_hits": EVI_MIN_KEYWORD_HITS,
            "min_score": EVI_MIN_SCORE,
            "min_experience_hits": EVI_MIN_EXPERIENCE_HITS,
            "hashtag_cut": EVI_HASHTAG_CUT,
            "allow_domains": EVI_ALLOW_DOMAINS,
            "ban_keywords": EVI_BAN_KEYWORDS,
            "template_min_chars": EVI_TEMPLATE_MIN_CHARS,
            "template_phrases": EVI_TEMPLATE_PHRASES,
            "lowinfo_list_words": EVI_LOWINFO_LIST_WORDS,
            "hashtag_soft_cut": EVI_HASHTAG_SOFT_CUT,
            "experience_words": EVI_EXPERIENCE_WORDS,
        },
    }
    with open(EVIDENCE_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(idx_obj, f, ensure_ascii=False, indent=2)

    print(f"[DONE] evidence kept={kept} miss={miss} unique_places={len(seen_place_keys)}")
    print(f"[DONE] wrote: {os.path.abspath(OUT_JSONL)}")
    print(f"[DONE] wrote: {os.path.abspath(EVIDENCE_INDEX_JSON)}")


if __name__ == "__main__":
    main()

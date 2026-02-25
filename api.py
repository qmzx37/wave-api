# api.py  (SINGLE FILE) — PATCHED FULL (timeout-safe)
# ✅ obs14 + diversity sampling + top5 debug + anti-collapse + action logging + RAG endpoints
# ✅ Kakao-only
# ✅ “형용사/작업/감성/뷰/조용” 들어와도 카카오 쿼리 절대 망가지지 않게: 카카오 쿼리 = (지역 + 카페/타입)만 사용
# ✅ “별침대” 영구 차단: 후보 풀/Top5/캐시 모두에서 필터링
# ✅ FI: Streamlit ReadTimeout 방지
#   - X요청 전체 시간예산(REQ_BUDGET_SEC) + 데드라인 초과 시 RAG/무거운 작업 자동 스킵
#   - Kakao requests timeout 튜닝(timeout=(3,5)) + 페이지 수 환경변수화(KAKAO_MAX_PAGES)
#   - RAG 중복 호출 제거(힌트 리랭크에서 받은 hits 재사용) → 같은 요청에서 rag_search 2번 돌던 문제 제거
#   - RAG k_each 완화(*10 → *3)
# ============================================================

from __future__ import annotations

import os
# ✅ .env 로드 (os.getenv 호출보다 반드시 먼저)
from dotenv import load_dotenv
load_dotenv()

import json
import datetime
import glob
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque
from fastapi.middleware.cors import CORSMiddleware
import re
import html
import unicodedata
import time
import random

import requests
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# ✅ RAG
from rag.crawl_url import crawl_once
from rag.build_index import build_index
from rag.search import search as rag_search  # ✅ retrieve 대신 search로 통일

# ✅ LLM

# ✅ Action space
from action_space import action_pair, NUM_ACTIONS

# ✅ Response builder
from response_builder import policy_v2_friend_with_action

# ✅ RLlib restore (old stack 고정)
try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
except Exception:
    ray = None  # type: ignore
    PPOConfig = None  # type: ignore

# ✅ 학습 때 쓴 env
try:
    from offline_env import NoiEOfflineEnv
except Exception:
    NoiEOfflineEnv = None  # type: ignore

def env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

# ✅ FIX: env_bool 정의 이후에 선언해야 NameError 안 남
LITE_MODE = env_bool("LITE_MODE", "0")

if not LITE_MODE:
    from infer_lora_8axis_final import load_model, infer_with_retries, append_log, KEYS
else:
    load_model = None
    infer_with_retries = None
    append_log = None
    KEYS = ("F","A","D","J","C","G","T","R")  

# ============================================================
# 🔎 Runtime self-check
# ============================================================
API_VERSION = "0.4.6-kakao-only-timeout-safe-singlefile"
HAS_DEBUG_RAG = True
SERVER_STARTED_AT: Optional[str] = None

# ✅ PPO ckpt debug info
RL_CKPT_INFO: Dict[str, Any] = {"source": None, "runs_dir": None, "picked": None, "env": None}

# ✅ TOP5 안정 캐시(20초 고정)
_TOP5_CACHE: Dict[str, Dict[str, Any]] = {}
_TOP5_CACHE_MAX = 512  # 메모리 보호
TOP5_STABLE_SEC_DEFAULT = 20


# ============================================================
# Request / Response
# ============================================================
class ChatRequest(BaseModel):
    text: str
    window: Optional[int] = 10
    avoid_franchise: Optional[bool] = None
    stable_top5_sec: Optional[int] = None
    balance_types: Optional[bool] = None
    # ✅ NEW: 프론트에서 지역/타입 강제 전달
    area: Optional[str] = None        # 예: "기장"
    must_area: Optional[str] = None   # 예: "기장" (더 강한 강제)
    cafe_type: Optional[str] = None   # 예: "뷰카페" | "베이커리" | "디저트" | "대형카페" | "카페"


class ChatResponse(BaseModel):
    ok: bool
    axes: Dict[str, float]
    wave: Dict[str, Any]
    action_id: int
    goal: str
    style: str
    reply: str
    debug: Dict[str, Any]


class RagCrawlRequest(BaseModel):
    url: str


class RagBuildRequest(BaseModel):
    model_name: Optional[str] = None


# ============================================================
# Utils
# ============================================================
def clamp01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))

def _clean_str(x: Any) -> Optional[str]:
    """
    Swagger/프론트에서 기본값으로 들어오는 'string' 같은 쓰레기 값을 None 처리.
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"string", "null", "none", "undefined"}:
        return None
    return s

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"</?b>", "", s, flags=re.IGNORECASE)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_axes(axes: Dict[str, Any]) -> Dict[str, float]:
    return {k: clamp01(axes.get(k, 0.0)) for k in KEYS}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


# ============================================================
# ✅ PPO checkpoint auto-discovery (Windows friendly)
# ============================================================
def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def find_latest_checkpoint(runs_dir: str) -> Optional[str]:
    runs_dir = (runs_dir or "").strip().strip('"').strip("'")
    if not runs_dir or (not os.path.exists(runs_dir)):
        return None

    pattern = os.path.join(runs_dir, "**", "checkpoint_*")
    cands = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    if not cands:
        return None

    cands.sort(key=_safe_mtime, reverse=True)
    return cands[0]


def resolve_rl_ckpt() -> Tuple[Optional[str], Dict[str, Any]]:
    """
    우선순위:
      1) env RL_CKPT (파일/디렉토리 존재)
      2) PPO_RUNS_DIR 아래에서 가장 최신 checkpoint_* 디렉토리
      3) 없으면 None
    """
    info: Dict[str, Any] = {"source": None, "runs_dir": None, "picked": None, "env": None}

    env_ckpt = (os.environ.get("RL_CKPT", "") or "").strip().strip('"').strip("'")
    info["env"] = env_ckpt

    # 1) env 지정 체크포인트가 존재하면 그대로 사용
    if env_ckpt and os.path.exists(env_ckpt):
        info["source"] = "env"
        info["picked"] = env_ckpt
        return env_ckpt, info

    # 2) runs_dir auto-discovery
    #    - Windows 개발 기본값을 네가 쓰는 경로로 잡아줌
    runs_dir = (os.environ.get("PPO_RUNS_DIR", r"C:\llm\train\runs") or "").strip().strip('"').strip("'")
    info["runs_dir"] = runs_dir

    latest = find_latest_checkpoint(runs_dir)
    if latest and os.path.exists(latest):
        info["source"] = "auto"
        info["picked"] = latest
        return latest, info

    info["source"] = "none"
    return None, info


# ============================================================
# ✅ Text normalization + mojibake fix
# ============================================================
def _maybe_fix_mojibake(s: str) -> str:
    s = s or ""
    if re.search(r"[가-힣]", s):
        return s
    if re.search(r"[ìëêðãÃ¢Â€]", s):
        try:
            fixed = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore").strip()
            if fixed and re.search(r"[가-힣]", fixed):
                return fixed
        except Exception:
            pass
    return s


def _normalize_user_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _maybe_fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_cache_key(s: str) -> str:
    s = _normalize_user_text(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _cache_prune_if_needed():
    global _TOP5_CACHE
    if len(_TOP5_CACHE) <= _TOP5_CACHE_MAX:
        return
    items = list(_TOP5_CACHE.items())
    items.sort(key=lambda kv: float(kv[1].get("ts", 0.0)))
    drop = max(1, len(items) - _TOP5_CACHE_MAX)
    for k, _ in items[:drop]:
        _TOP5_CACHE.pop(k, None)


# ============================================================
# ✅ BLOCKLIST: “별침대” 같은 이상한 결과 영구 차단
# ============================================================
BLOCKLIST_PATTERNS: List[re.Pattern] = [
    re.compile(r"별\s*침대", re.IGNORECASE),
]


def _is_blocklisted_place(p: Dict[str, Any]) -> bool:
    hay = " ".join(
        [
            str(p.get("place_name", "") or ""),
            str(p.get("category_name", "") or ""),
            str(p.get("address", "") or ""),
            str(p.get("road_address_name", "") or ""),
            str(p.get("address_name", "") or ""),
            str(p.get("place_url", "") or ""),
        ]
    )
    return any(rx.search(hay) for rx in BLOCKLIST_PATTERNS)


def _filter_blocklist(places: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out = []
    n_drop = 0
    for p in places or []:
        if _is_blocklisted_place(p):
            n_drop += 1
            continue
        out.append(p)
    return out, n_drop


# ============================================================
# ✅ RAG 힌트 스코어링 (조용함/콘센트/테이블)
# ============================================================
_HINT_KEYS = ("quiet", "outlet", "table")

_HINT_PATTERNS: Dict[str, List[str]] = {
    "quiet": [
        "조용",
        "quiet",
        "정숙",
        "차분",
        "잔잔",
        "아늑",
        "스터디",
        "공부",
        "집중",
        "노트북",
        "대화 적",
        "소음 적",
        "시끄럽지",
        "분위기 좋",
    ],
    "outlet": [
        "콘센트",
        "전원",
        "충전",
        "충전기",
        "멀티탭",
        "outlet",
        "plug",
        "socket",
        "USB",
        "usb",
        "노트북 충전",
        "전기",
    ],
    "table": [
        "테이블",
        "자리 넓",
        "좌석 넓",
        "큰 테이블",
        "넓은 테이블",
        "작업",
        "노트북",
        "1인석",
        "2인석",
        "4인석",
        "좌석",
        "의자",
        "테이블 간격",
    ],
}


def _detect_hint_weights(user_text: str) -> Dict[str, float]:
    t = (user_text or "").lower()
    w = {"quiet": 0.0, "outlet": 0.0, "table": 0.0}
    if any(k in t for k in ["조용", "quiet", "스터디", "공부", "집중", "노트북 하기", "작업하기", "작업"]):
        w["quiet"] = 1.0
    if any(k in t for k in ["콘센트", "충전", "전원", "노트북 충전", "usb", "USB"]):
        w["outlet"] = 1.0
    if any(k in t for k in ["테이블", "좌석", "자리", "넓은", "큰 테이블", "작업", "노트북"]):
        w["table"] = 1.0
    return w


def _hit_text_fields(hit: Dict[str, Any]) -> str:
    if not isinstance(hit, dict):
        return ""
    parts: List[str] = []
    for key in ("text", "content", "snippet", "title", "desc", "description", "summary"):
        v = hit.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    u = hit.get("url")
    if isinstance(u, str) and u.strip():
        parts.append(u.strip())
    return _clean_text(" ".join(parts))


def score_hints_from_rag_hits(
    hits: List[Dict[str, Any]],
    *,
    hint_weights: Dict[str, float],
) -> Tuple[float, Dict[str, Any]]:
    hw = {k: float(hint_weights.get(k, 0.0)) for k in _HINT_KEYS}
    if sum(hw.values()) <= 0.0:
        return 0.0, {"weights": hw, "counts": {"quiet": 0, "outlet": 0, "table": 0}, "rag_score_sum": 0.0}

    counts = {"quiet": 0, "outlet": 0, "table": 0}
    rag_score_sum = 0.0

    joined = " ".join(_hit_text_fields(h) for h in (hits or []))
    joined_l = joined.lower()

    for hk in _HINT_KEYS:
        c = 0
        for pat in _HINT_PATTERNS.get(hk, []):
            try:
                c += joined_l.count(pat.lower())
            except Exception:
                pass
        counts[hk] = int(c)

    for h in (hits or []):
        try:
            rag_score_sum += float(h.get("score", 0.0) or 0.0)
        except Exception:
            pass

    def sat(x: float) -> float:
        return float(1.0 - np.exp(-max(0.0, x)))

    quiet_s = sat(counts["quiet"] / 2.0)
    outlet_s = sat(counts["outlet"] / 2.0)
    table_s = sat(counts["table"] / 2.0)
    rag_s = sat(np.log1p(max(0.0, rag_score_sum)))

    hint_core = (hw["quiet"] * quiet_s + hw["outlet"] * outlet_s + hw["table"] * table_s) / max(1e-6, sum(hw.values()))
    score = float(np.clip(0.85 * hint_core + 0.15 * rag_s, 0.0, 1.0))

    dbg = {
        "weights": hw,
        "counts": counts,
        "quiet_s": float(quiet_s),
        "outlet_s": float(outlet_s),
        "table_s": float(table_s),
        "rag_score_sum": float(rag_score_sum),
        "rag_s": float(rag_s),
        "score": float(score),
    }
    return score, dbg


def rerank_top5_by_hints(
    places_top5: List[Dict[str, Any]],
    hint_scores: Dict[str, float],
    *,
    alpha: float = 0.35,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    scored = []
    for i, p in enumerate(places_top5 or []):
        pid = str(p.get("id", "") or "").strip()
        hs = float(hint_scores.get(pid, 0.0))
        base = 1.0 / float(i + 1)
        final = base + alpha * hs
        scored.append((final, base, hs, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    out = [x[3] for x in scored]
    dbg = {
        "alpha": float(alpha),
        "rows": [
            {
                "id": str(x[3].get("id", "") or ""),
                "name": str(x[3].get("place_name", "") or ""),
                "base_rank": float(x[1]),
                "hint_score": float(x[2]),
                "final": float(x[0]),
            }
            for x in scored
        ],
    }
    return out, dbg


# ============================================================
# ✅ Kakao Places
# ============================================================
KAKAO_REST_API_KEY = (os.getenv("KAKAO_REST_API_KEY", "") or os.getenv("KAKAO_REST_KEY", "")).strip()
# (선택) 서버 켜질 때 콘솔에서 키 로드 여부 확인용
print(
    "KAKAO KEY LOADED:",
    bool(KAKAO_REST_API_KEY),
    (KAKAO_REST_API_KEY[:6] + "***") if KAKAO_REST_API_KEY else "",
)
_KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"

AREA_BIAS = {
    "광안리": {"x": 129.118, "y": 35.153, "radius": 3500},
    "해운대": {"x": 129.158, "y": 35.163, "radius": 4500},
    "서면": {"x": 129.058, "y": 35.157, "radius": 3000},
    "남포": {"x": 129.035, "y": 35.098, "radius": 3000},
    "전포": {"x": 129.063, "y": 35.155, "radius": 2500},
}

FRANCHISE_BRANDS = [
    "스타벅스",
    "투썸",
    "투썸플레이스",
    "이디야",
    "메가커피",
    "컴포즈",
    "더벤티",
    "할리스",
    "파스쿠찌",
    "커피빈",
    "엔제리너스",
    "폴바셋",
    "탐앤탐스",
    "카페베네",
    "빽다방",
    "매머드",
    "매머드커피",
    "드롭탑",
    "달콤커피",
]


# ============================================================
# ✅ NEW: 타입별 균형 샘플링
# ============================================================
TYPE_BUCKETS = ("viewcafe", "bakery", "dessert", "general")

_TYPE_KEYWORDS = {
    "viewcafe": [
        "오션뷰",
        "바다뷰",
        "전망",
        "루프탑",
        "테라스",
        "정원",
        "잔디",
        "파노라마",
        "리버뷰",
        "마운틴뷰",
        "브릿지뷰",
        "광안대교",
    ],
    "bakery": ["베이커리", "빵", "제과", "브레드", "크루아상", "크로와상", "휘낭시에", "스콘", "식빵", "바게트"],
    "dessert": ["디저트", "케이크", "타르트", "마카롱", "쿠키", "젤라또", "푸딩", "파르페", "아이스크림"],
}


def _place_text_for_type(p: Dict[str, Any]) -> str:
    name = str(p.get("place_name", "") or "")
    cat = str(p.get("category_name", "") or "")
    extra = str(p.get("road_address_name", "") or "") + " " + str(p.get("address_name", "") or "")
    return (name + " " + cat + " " + extra).strip().lower()


def _detect_place_type_bucket(p: Dict[str, Any]) -> str:
    t = _place_text_for_type(p)
    if any(k.lower() in t for k in _TYPE_KEYWORDS["viewcafe"]):
        return "viewcafe"
    if any(k.lower() in t for k in _TYPE_KEYWORDS["bakery"]):
        return "bakery"
    if any(k.lower() in t for k in _TYPE_KEYWORDS["dessert"]):
        return "dessert"
    return "general"


def _group_by_type(candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in TYPE_BUCKETS}
    for p in candidates or []:
        b = _detect_place_type_bucket(p)
        buckets[b].append(p)
    return buckets


def _wants_cafe_only(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    keys = ["카페", "커피", "coffee", "베이커리", "디저트", "찻집", "브런치"]
    qq = q.lower()
    return any(k.lower() in qq for k in keys)


def _keep_by_category(place: Dict[str, Any], want_cafe_only: bool) -> bool:
    if not want_cafe_only:
        return True
    cat = str(place.get("category_name", "") or "").strip()
    return "카페" in cat


def _is_franchise_name(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    for b in FRANCHISE_BRANDS:
        if b in n:
            return True
    return False


def _brand_key(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    for b in FRANCHISE_BRANDS:
        if b in n:
            return b
    n2 = re.sub(r"[\(\)\[\]\{\}\-_/|·•:]", " ", n)
    n2 = re.sub(r"\s+", " ", n2).strip()
    return (n2.split(" ")[0] if n2 else n)[:10]


def _dedupe_places(places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in places or []:
        pid = str(p.get("id", "") or "").strip()
        name = str(p.get("place_name", "") or "").strip()
        key = pid or name
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _addr_text(p: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(p.get("address", "") or ""),
            str(p.get("road_address_name", "") or ""),
            str(p.get("address_name", "") or ""),
        ]
    ).strip()

# ✅ 부산 지역 별칭표 (주소에 실제로 찍히는 행정/동/역/해변명까지 포함)
AREA_MATCH_ALIASES = {
    # 해변/관광권역
    "광안리": ["광안리", "광안리해수욕장", "수영구", "광안동", "민락동", "남천동", "금련산", "금련산역", "광안역"],
    "해운대": ["해운대", "해운대해수욕장", "해운대구", "우동", "중동", "좌동", "재송동", "송정", "송정해수욕장", "마린시티", "장산", "장산역", "중동역", "해운대역", "센텀", "센텀시티", "벡스코", "동백", "동백섬", "청사포", "미포"],
    "송도": ["송도", "송도해수욕장", "서구", "암남동", "남부민동", "송도해상케이블카"],
    "다대포": ["다대포", "다대포해수욕장", "사하구", "다대동"],
    "태종대": ["태종대", "영도구", "동삼동", "태종대유원지"],
    "오륙도": ["오륙도", "남구", "용호동", "이기대", "이기대공원"],

    # 도심권역
    "서면": ["서면", "부산진구", "부전동", "범천동", "전포동", "부암동", "가야동", "서면역", "부전역", "전포역"],
    "전포": ["전포", "전포동", "부산진구", "전포역"],
    "남포": ["남포", "남포동", "중구", "광복동", "동광동", "부평동", "신창동", "대청동", "보수동", "자갈치", "자갈치역", "남포역", "국제시장", "광복로"],
    "부산역": ["부산역", "동구", "초량동", "범일동", "좌천동", "부산진역", "부산진역"],
    "광복": ["광복", "광복동", "중구", "광복로"],

    # 동부권/근교
    "기장": ["기장", "기장군", "기장읍", "일광", "일광읍", "정관", "정관읍", "장안", "장안읍", "철마", "칠암", "송정"],

    # 북구/서구/사상권
    "사상": ["사상", "사상구", "괘법동", "감전동", "학장동", "엄궁동", "덕포동", "사상역", "괘법르네시떼역"],
    "덕천": ["덕천", "북구", "덕천동", "구포동", "만덕동", "화명동", "덕천역", "구포역"],
    "하단": ["하단", "사하구", "하단동", "당리동", "괴정동", "신평동", "장림동", "하단역"],
    "강서": ["강서구", "명지", "명지동", "대저", "대저동", "가덕도", "신호동"],

    # 남구/연제구/동래권
    "경성대": ["경성대", "남구", "대연동", "용호동", "경성대·부경대역", "부경대"],
    "연산동": ["연산", "연산동", "연제구", "연산역", "거제역", "시청", "시청역"],
    "동래": ["동래", "동래구", "명륜동", "온천동", "사직동", "안락동", "수안동", "동래역", "온천장역", "미남역", "사직역"],
}

def _area_match(p: Dict[str, Any], must_area: str) -> bool:
    a = (must_area or "").strip()
    if not a:
        return True

    addr = _addr_text(p)

    # ✅ 1) 기존 직접 포함(빠른 통과)
    if a in addr:
        return True
    if (a + "구") in addr:
        return True
    if (a + "군") in addr:
        return True

    # ✅ 2) 별칭 매칭(핵심)
    aliases = AREA_MATCH_ALIASES.get(a, [])
    return any(k in addr for k in aliases)



# ============================================================
# ✅ SAFE QUERY BUILDER (핵심 패치)
# - 카카오 쿼리는 "지역 + 카테고리(카페/베이커리/디저트/뷰카페)"만 사용
# - 사용자 문장(조용/감성/작업/뷰/핫플/추천해줘 등)은 절대 카카오 쿼리에 섞지 않는다.
# ============================================================
REQUEST_NOISE_RX = re.compile(
    r"""
    (?:추천|알려)\s*(?:해|해줘|해줘요|해주세요|해주라|해줘라|해줄래|해주실래)\b
    |(?:찾아)\s*(?:줘|주라|주세요)\b
    |(?:좀|조금|혹시|그냥|아무거나|제발)\b
    |(?:베스트|best|top)\s*\d+\b
    |(?:진짜|핫한|유명한)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ADJ_NOISE_RX = re.compile(
    r"(조용|조용한|감성|분위기|뷰|오션뷰|바다뷰|바다\s*뷰|작업|노트북|콘센트|충전|테이블|좌석|자리|넓은|데이트|힙|인스타|사진|포토|예쁜|깔끔)",
    re.IGNORECASE,
)


def _normalize_for_intent(s: str) -> str:
    s = (s or "").strip()
    s = REQUEST_NOISE_RX.sub(" ", s)
    s = ADJ_NOISE_RX.sub(" ", s)
    s = re.sub(r"[?!.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_kakao_safe_query(*, area: str, cafe_type: str) -> str:
    a = (area or "").strip()
    t = (cafe_type or "카페").strip()

    # 타입 normalize
    if t == "뷰카페":
        t2 = "오션뷰 카페"
    elif t == "대형카페":
        t2 = "대형 카페"
    elif t == "베이커리":
        t2 = "베이커리 카페"
    elif t == "디저트":
        t2 = "디저트 카페"
    else:
        t2 = "카페"

    if a:
        return f"부산 {a} {t2}".strip()
    return f"부산 {t2}".strip()


def build_kakao_fallback_queries(*, area: str, cafe_type: str) -> List[str]:
    primary = build_kakao_safe_query(area=area, cafe_type=cafe_type)
    a = (area or "").strip()

    if cafe_type == "뷰카페":
        base2 = "오션뷰 카페"
    elif cafe_type == "대형카페":
        base2 = "대형 카페"
    elif cafe_type == "베이커리":
        base2 = "베이커리 카페"
    elif cafe_type == "디저트":
        base2 = "디저트 카페"
    else:
        base2 = "카페"

    cands = []
    cands.append(primary)
    if a:
        cands.append(f"{a} {base2}")
        cands.append(f"부산 {a} 카페")
        cands.append(f"{a} 카페")
        cands.append("부산 수영구 카페")  # 안전망
    cands.append("부산 카페")

    seen = set()
    out = []
    for q in cands:
        q = re.sub(r"\s+", " ", (q or "").strip())
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def kakao_place_search_multi(
    query: str,
    *,
    size_per_page: int = 15,
    want_total: int = 30,
    area: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    q = (query or "").strip()
    want_cafe_only = _wants_cafe_only(q)

    dbg: Dict[str, Any] = {
        "ok": False,
        "status": None,
        "error": None,
        "query": q,
        "used_query": q,
        "fallback_tried": [],
        "area": (area or "").strip(),
        "used_bias": False,
        "count": 0,
        "has_key": bool(KAKAO_REST_API_KEY),
        "key_prefix": (KAKAO_REST_API_KEY[:6] + "***") if KAKAO_REST_API_KEY else "",
        "want_cafe_only": bool(want_cafe_only),
        "filtered_out": 0,
        "filtered_out_franchise": 0,
        "filtered_blocklist": 0,
        "pages": [],
        "max_pages": int(env_int("KAKAO_MAX_PAGES", 2)),
    }

    if not q:
        dbg["error"] = "empty_query"
        return [], dbg

    if not KAKAO_REST_API_KEY:
        dbg["error"] = "missing_kakao_key"
        return [], dbg

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    size = max(1, min(int(size_per_page), 15))
    want_total = max(1, min(int(want_total), 30))

    used_bias = False
    params_bias: Dict[str, Any] = {}
    if env_bool("KAKAO_USE_BIAS", "0"):
        bias = AREA_BIAS.get((area or "").strip())
        if bias:
            params_bias["x"] = bias["x"]
            params_bias["y"] = bias["y"]
            params_bias["radius"] = bias["radius"]
            used_bias = True
    dbg["used_bias"] = bool(used_bias)

    # ✅ 이 함수는 "단일 query"만 수행. (fallback은 바깥에서 처리)
    fallback_queries = [q]
    dbg["fallback_tried"] = list(fallback_queries)

    def _search_once(one_q: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params_base: Dict[str, Any] = {"query": one_q, "size": size}
        params_base.update(params_bias)

        out_all: List[Dict[str, Any]] = []
        filtered_out = 0
        filtered_block = 0
        pages_dbg = []

        max_pages = int(env_int("KAKAO_MAX_PAGES", 2))
        max_pages = max(1, min(max_pages, 3))

        for page in range(1, max_pages + 1):
            if len(out_all) >= want_total:
                break

            params = dict(params_base)
            params["page"] = page

            page_dbg = {
                "page": page,
                "status": None,
                "error": None,
                "raw_docs": 0,
                "kept": 0,
                "filtered_out": 0,
                "filtered_blocklist": 0,
                "meta": None,
                "keys": [],
            }

            try:
                # ✅ connect 3초 / read 5초
                r = requests.get(_KAKAO_KEYWORD_URL, headers=headers, params=params, timeout=(3, 5))
                page_dbg["status"] = int(r.status_code)
            except Exception as e:
                page_dbg["error"] = f"request_error:{repr(e)}"
                pages_dbg.append(page_dbg)
                return out_all[:want_total], {"pages": pages_dbg, "error": page_dbg["error"], "status": None, "count": len(out_all)}

            if r.status_code != 200:
                try:
                    page_dbg["error"] = f"bad_status:{int(r.status_code)}:{(r.text or '')[:200]}"
                except Exception:
                    page_dbg["error"] = f"bad_status:{int(r.status_code)}"
                pages_dbg.append(page_dbg)
                return out_all[:want_total], {"pages": pages_dbg, "error": page_dbg["error"], "status": int(r.status_code), "count": len(out_all)}

            try:
                data = r.json()
                page_dbg["keys"] = list(data.keys()) if isinstance(data, dict) else []
                if isinstance(data, dict) and "meta" in data:
                    page_dbg["meta"] = data.get("meta")
            except Exception as e:
                page_dbg["error"] = f"json_error:{repr(e)}"
                pages_dbg.append(page_dbg)
                return out_all[:want_total], {"pages": pages_dbg, "error": page_dbg["error"], "status": int(r.status_code), "count": len(out_all)}

            docs = (data.get("documents") or []) if isinstance(data, dict) else []
            page_dbg["raw_docs"] = int(len(docs))

            if not docs:
                pages_dbg.append(page_dbg)
                break

            for d in docs:
                road = str(d.get("road_address_name", "") or "").strip()
                addr = str(d.get("address_name", "") or "").strip()
                address = road if road else addr

                place = {
                    "id": str(d.get("id", "") or "").strip(),
                    "place_name": str(d.get("place_name", "") or "").strip(),
                    "address_name": addr,
                    "road_address_name": road,
                    "address": address,
                    "phone": str(d.get("phone", "") or "").strip(),
                    "place_url": str(d.get("place_url", "") or "").strip(),
                    "category_name": str(d.get("category_name", "") or "").strip(),
                    "x": str(d.get("x", "") or "").strip(),
                    "y": str(d.get("y", "") or "").strip(),
                }

                if _is_blocklisted_place(place):
                    filtered_block += 1
                    page_dbg["filtered_blocklist"] = int(page_dbg["filtered_blocklist"]) + 1
                    continue

                if not _keep_by_category(place, want_cafe_only=want_cafe_only):
                    filtered_out += 1
                    page_dbg["filtered_out"] = int(page_dbg["filtered_out"]) + 1
                    continue

                out_all.append(place)
                page_dbg["kept"] = int(page_dbg["kept"]) + 1

                if len(out_all) >= want_total:
                    break

            pages_dbg.append(page_dbg)

        out_all2 = _dedupe_places(out_all)

        out_all2, drop2 = _filter_blocklist(out_all2)
        filtered_block += int(drop2)

        status0 = int(pages_dbg[0]["status"]) if pages_dbg else 200
        return out_all2[:want_total], {
            "pages": pages_dbg,
            "status": status0,
            "error": None,
            "count": len(out_all2[:want_total]),
            "filtered_out": int(filtered_out),
            "filtered_blocklist": int(filtered_block),
        }

    final_places: List[Dict[str, Any]] = []
    final_pages = []
    final_status = None
    final_error = None
    final_filtered_out = 0
    final_filtered_block = 0
    used_query = q

    for one_q in fallback_queries:
        places, one_dbg = _search_once(one_q)
        final_pages = one_dbg.get("pages", [])
        final_status = one_dbg.get("status")
        final_error = one_dbg.get("error")
        final_filtered_out = int(one_dbg.get("filtered_out", 0) or 0)
        final_filtered_block = int(one_dbg.get("filtered_blocklist", 0) or 0)

        if places:
            final_places = places
            used_query = one_q
            final_error = None
            break

    dbg["used_query"] = used_query
    dbg["pages"] = final_pages
    dbg["status"] = final_status
    dbg["error"] = final_error
    dbg["filtered_out"] = int(final_filtered_out)
    dbg["filtered_blocklist"] = int(final_filtered_block)

    # ✅ 성공/실패를 디버그에서 일관되게
    dbg["ok"] = bool(final_places) and (final_error is None)
    dbg["count"] = int(len(final_places[:want_total]))

    return final_places[:want_total], dbg

def _weighted_sample_without_replacement(
    items: List[Dict[str, Any]],
    weights: List[float],
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    items = list(items)
    weights = list(weights)
    out: List[Dict[str, Any]] = []
    for _ in range(min(k, len(items))):
        tot = sum(max(0.0, w) for w in weights)
        if tot <= 0:
            idx = rng.randrange(0, len(items))
        else:
            r = rng.random() * tot
            acc = 0.0
            idx = 0
            for i, w in enumerate(weights):
                acc += max(0.0, w)
                if acc >= r:
                    idx = i
                    break
        out.append(items.pop(idx))
        weights.pop(idx)
    return out


def pick_top5_diverse(
    candidates: List[Dict[str, Any]],
    *,
    avoid_franchise: bool,
    stable_seed_bucket: Optional[int] = None,
    balance_types: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dbg = {
        "avoid_franchise": bool(avoid_franchise),
        "filtered_franchise": 0,
        "filtered_blocklist": 0,
        "strategy": "weighted+brand_diverse+type_balance" if balance_types else "weighted+brand_diverse",
        "balance_types": bool(balance_types),
        "picked_by_type": {},
    }

    pool = _dedupe_places(candidates or [])
    if not pool:
        return [], dbg

    pool, drop_b = _filter_blocklist(pool)
    dbg["filtered_blocklist"] = int(drop_b)
    if not pool:
        return [], dbg

    filtered_franchise = 0
    if avoid_franchise:
        keep = []
        for p in pool:
            name = str(p.get("place_name", "") or "")
            if _is_franchise_name(name):
                filtered_franchise += 1
                continue
            keep.append(p)
        if len(keep) >= 5:
            pool = keep
        else:
            pool = keep + [p for p in pool if p not in keep]
    dbg["filtered_franchise"] = int(filtered_franchise)

    rng = random.Random()
    if stable_seed_bucket is None:
        stable_seed_bucket = int(time.time() // 30)
    rng.seed(stable_seed_bucket)

    weights = [1.0 / float(i + 1) for i in range(len(pool))]
    warm = _weighted_sample_without_replacement(pool, weights, k=min(12, len(pool)), rng=rng)

    used_brand = set()
    picked: List[Dict[str, Any]] = []

    def try_add(p: Dict[str, Any], used_brand: set) -> bool:
        bk = _brand_key(str(p.get("place_name", "") or ""))
        if bk and bk in used_brand:
            return False
        picked.append(p)
        if bk:
            used_brand.add(bk)
        return True

    if balance_types:
        buckets = _group_by_type(warm)
        order = ["viewcafe", "bakery", "dessert", "general"]

        for b in order:
            if len(picked) >= 5:
                break
            for p in buckets.get(b, []):
                if len(picked) >= 5:
                    break
                if try_add(p, used_brand):
                    dbg["picked_by_type"][b] = dbg["picked_by_type"].get(b, 0) + 1
                    break

        if len(picked) < 5:
            for p in warm:
                if len(picked) >= 5:
                    break
                if p in picked:
                    continue
                if try_add(p, used_brand):
                    b = _detect_place_type_bucket(p)
                    dbg["picked_by_type"][b] = dbg["picked_by_type"].get(b, 0) + 1

    if (not balance_types) and len(picked) < 5:
        for p in warm:
            if len(picked) >= 5:
                break
            if p in picked:
                continue
            if try_add(p, used_brand):
                pass

    if len(picked) < 5:
        for p in pool:
            if len(picked) >= 5:
                break
            if p in picked:
                continue
            bk = _brand_key(str(p.get("place_name", "") or ""))
            if bk and bk in used_brand:
                continue
            picked.append(p)
            if bk:
                used_brand.add(bk)
            if balance_types:
                b = _detect_place_type_bucket(p)
                dbg["picked_by_type"][b] = dbg["picked_by_type"].get(b, 0) + 1

    if len(picked) < 5:
        for p in pool:
            if len(picked) >= 5:
                break
            if p in picked:
                continue
            picked.append(p)
            if balance_types:
                b = _detect_place_type_bucket(p)
                dbg["picked_by_type"][b] = dbg["picked_by_type"].get(b, 0) + 1

    picked, drop2 = _filter_blocklist(picked)
    dbg["filtered_blocklist"] = int(dbg["filtered_blocklist"]) + int(drop2)

    return picked[:5], dbg


def pick_places_simple(places: List[Dict[str, Any]], *, want: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in (places or [])[: max(1, int(want))]:
        out.append(
            {
                "id": str(p.get("id", "") or "").strip(),
                "place_name": str(p.get("place_name", "") or "").strip(),
                "address": str(p.get("address", "") or "").strip()
                or (
                    str(p.get("road_address_name", "") or "").strip()
                    or str(p.get("address_name", "") or "").strip()
                ),
                "phone": str(p.get("phone", "") or "").strip(),
                "place_url": str(p.get("place_url", "") or "").strip(),
            }
        )
    return out


# ============================================================
# ✅ AMPLIFY 강제
# ============================================================
AMPLIFY_KEYWORDS = [
    "디저트",
    "카페",
    "베이커리",
    "빵집",
    "찻집",
    "젤라또",
    "맛집",
    "밥집",
    "술집",
    "데이트",
    "놀거리",
    "전시",
    "핫플",
    "광안리",
    "해운대",
    "서면",
    "남포",
    "센텀",
    "송도",
    "기장",
    "연산동",
    "전포",
    "수영",
    "민락",
    "송정",
    "정관",
    "사상",
    "동래",
]
AMPLIFY_ACTION_IDS = {9, 10, 11}


def _should_force_amplify(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(k.lower() in t for k in AMPLIFY_KEYWORDS)


# ============================================================
# ✅ user_text -> (area/type) 자동 추출
# ============================================================
AREA_ALIASES: List[Tuple[str, List[str]]] = [
    ("기장", ["기장", "기장군", "일광", "일광읍", "정관", "정관읍", "장안", "장안읍", "철마", "송정"]),
    ("광안리", ["광안리", "수영", "민락", "금련산역"]),
    ("해운대", ["해운대", "센텀", "센텀시티", "우동", "중동", "장산역", "재송"]),
    ("전포", ["전포", "전포동"]),
    ("서면", ["서면", "부전", "범천", "전포역", "서면역"]),
]

_OCEAN_VIEW_KEYS = [
    "오션뷰",
    "바다뷰",
    "해변뷰",
    "해안뷰",
    "sea view",
    "ocean view",
    "해수욕장",
    "파도",
    "바닷가",
    "해변",
    "바다",
    "광안대교",
    "마린시티",
    "더베이",
    "동백섬",
    "청사포",
    "송정",
    "기장",
]


def _is_ocean_view_intent(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(k.lower() in t for k in _OCEAN_VIEW_KEYS)


TYPE_ALIASES: List[Tuple[str, List[str]]] = [
    ("베이커리", ["베이커리", "빵집", "제과", "빵", "브레드", "크로와상", "휘낭시에"]),
    ("뷰카페", ["오션뷰", "바다뷰", "해변뷰", "해안뷰", "sea view", "ocean view", "해수욕장", "광안대교", "마린시티"]),
    ("대형카페", ["대형", "넓은", "루프탑", "정원", "잔디", "테라스"]),
    ("디저트", ["디저트", "쿠키", "케이크", "타르트", "마카롱", "젤라또", "푸딩", "아이스크림", "파르페"]),
]


def _detect_area_type_from_text(text: str) -> Dict[str, str]:
    t = (text or "").strip()
    if not t:
        return {}
    tt = re.sub(r"\s+", " ", t)
    out: Dict[str, str] = {}

    for canonical, keys in AREA_ALIASES:
        if any(k in tt for k in keys):
            out["area"] = canonical
            break

    if _is_ocean_view_intent(tt):
        out["type"] = "뷰카페"
        return out

    for canonical, keys in TYPE_ALIASES:
        if canonical == "뷰카페":
            continue
        if any(k in tt for k in keys):
            out["type"] = canonical
            break

    return out


# ============================================================
# UTF-8 JSON Response
# ============================================================
class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")


# ============================================================
# Global state
# ============================================================
app = FastAPI(
    title="NoiE Wave Chat API",
    version=API_VERSION,
    default_response_class=UTF8JSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOK = None
MODEL = None
LORA_OK = False

RL_ALGO = None
RL_READY = False
LAST_ACTIONS: Deque[int] = deque(maxlen=8)



@app.on_event("startup")
def _startup():
    global TOK, MODEL, LORA_OK, RL_ALGO, RL_READY, SERVER_STARTED_AT, RL_CKPT_INFO

    SERVER_STARTED_AT = datetime.datetime.now().isoformat()

    # ✅ LITE_MODE면 LLM 로딩 스킵
    if LITE_MODE:
        print("🚀 LITE_MODE=1: skip LLM/LoRA loading")
        TOK, MODEL, LORA_OK = None, None, False
    else:
        TOK, MODEL, LORA_OK = load_model()

    # (아래 PPO 부분은 그대로)


    if ray is None or PPOConfig is None or NoiEOfflineEnv is None:
        RL_ALGO = None
        RL_READY = False
        print("⚠️ PPO disabled: ray/PPOConfig/NoiEOfflineEnv not available.")
        print(f"   - ray={ray is not None}, PPOConfig={PPOConfig is not None}, NoiEOfflineEnv={NoiEOfflineEnv is not None}")
        return

    ckpt, ckpt_info = resolve_rl_ckpt()
    RL_CKPT_INFO = dict(ckpt_info or {})

    if not ckpt:
        RL_READY = False
        print("⚠️ PPO disabled: checkpoint not found.")
        print("   - env RL_CKPT =", ckpt_info.get("env"))
        print("   - runs_dir    =", ckpt_info.get("runs_dir"))
        return

    try:
        os.environ["RL_CKPT"] = ckpt
    except Exception:
        pass

    print(f"✅ PPO checkpoint ({ckpt_info.get('source')}): {ckpt}")

    try:
        if (ray is not None) and (not ray.is_initialized()):
            # ✅ Windows/로컬 서버에서 대시보드 불필요하면 끄는 게 안전
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        log_path = os.environ.get("LOG_PATH", "/data/emotion_log.jsonl")
        episode_len = int(os.environ.get("EPISODE_LEN", "20"))
        window = int(os.environ.get("ENV_WINDOW", "3000"))

        cfg = (
            PPOConfig()
            .environment(
                env=NoiEOfflineEnv,
                env_config={"log_path": log_path, "window": window, "episode_len": episode_len},
            )
            .framework("torch")
            .env_runners(num_env_runners=0)
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        )

        RL_ALGO = cfg.build()
        try:
            RL_ALGO.restore(ckpt)
        except Exception:
            cand_files = []
            try:
                cand_files = glob.glob(os.path.join(ckpt, "checkpoint_*")) + glob.glob(os.path.join(ckpt, "checkpoint"))
            except Exception:
                cand_files = []

            if cand_files:
                cand_files.sort(key=_safe_mtime, reverse=True)
                RL_ALGO.restore(cand_files[0])
            else:
                raise

        RL_READY = True
        print(f"🧠 PPO restored: {ckpt}")

    except Exception as e:
        RL_ALGO = None
        RL_READY = False
        print("⚠️ PPO load failed:", repr(e))
@app.get("/health")
def health():
    base = os.path.dirname(__file__)

    rag_store_dir = os.path.join(base, "rag", "store")
    rag_index = os.path.join(rag_store_dir, "index.faiss")
    rag_meta = os.path.join(rag_store_dir, "meta.jsonl")

    rag_ready = os.path.exists(rag_index) and os.path.exists(rag_meta)

    return {
        "ok": True,
        "api_version": API_VERSION,
        "api_file": __file__,
        "started_at": SERVER_STARTED_AT,
        "has_debug_rag": HAS_DEBUG_RAG,
        "rag_index_ready": bool(rag_ready),
        "rag_store_dir": rag_store_dir,
        "rag_index_path": rag_index,
        "rag_meta_path": rag_meta,
        "rl_ready": bool(RL_READY),
        "rl_ckpt": os.environ.get("RL_CKPT", ""),
        "ppo_runs_dir": os.environ.get("PPO_RUNS_DIR", r"C:\llm\train\runs"),
        "rl_ckpt_source": str((RL_CKPT_INFO or {}).get("source") or ""),
        "rl_ckpt_picked": str((RL_CKPT_INFO or {}).get("picked") or ""),
        "rl_ckpt_env": str((RL_CKPT_INFO or {}).get("env") or ""),
        "rl_runs_dir_effective": str((RL_CKPT_INFO or {}).get("runs_dir") or ""),
        "num_actions": int(NUM_ACTIONS),
        "lora_ok": bool(LORA_OK),
        "obs_dim": 14,
        "kakao_enabled": bool(KAKAO_REST_API_KEY),
        "kakao_use_bias": bool(env_bool("KAKAO_USE_BIAS", "0")),
        "naver_enabled": False,
        "top5_stable_sec": int(env_int("TOP5_STABLE_SEC", TOP5_STABLE_SEC_DEFAULT)),
        "franchise_filter_default": bool(env_bool("AVOID_FRANCHISE", "0")),
        "balance_types_default": bool(env_bool("BALANCE_TYPES", "1")),
        "rag_hint_rerank_alpha": float(env_float("RAG_HINT_ALPHA", 0.35)),
        "blocklist": [rx.pattern for rx in BLOCKLIST_PATTERNS],
        "req_budget_sec": float(os.environ.get("REQ_BUDGET_SEC", "25")),
        "kakao_max_pages": int(env_int("KAKAO_MAX_PAGES", 2)),
    }


# ============================================================
# ✅ RAG Endpoints
# ============================================================
@app.post("/rag/crawl")
def rag_crawl(req: RagCrawlRequest):
    url = (req.url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "url must start with http:// or https://"}

    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "rag", "data")

    try:
        saved = crawl_once(url, out_dir=data_dir)
        return {"ok": True, "url": url, "saved": saved}
    except Exception as e:
        return {"ok": False, "url": url, "error": repr(e)}


@app.post("/rag/build")
def rag_build(req: RagBuildRequest):
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "rag", "data")
    store_dir = os.path.join(base, "rag", "store")

    model_name = (req.model_name or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"

    try:
        result = build_index(
            data_dir=data_dir,
            store_dir=store_dir,
            model_name=model_name,
        )
        return result
    except Exception as e:
        return {"ok": False, "error": repr(e), "data_dir": data_dir, "store_dir": store_dir}


# ============================================================
# ✅ Main Chat Endpoint
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    LOG_PATH = os.environ.get("LOG_PATH", "/data/emotion_log.jsonl")
    user_text = _normalize_user_text(req.text)
    window = int(req.window or 10)

    # ============================================================
    # ✅ Request time budget (hard cap)
    # ============================================================
    REQ_BUDGET_SEC = float(os.environ.get("REQ_BUDGET_SEC", "25"))  # 기본 25초 추천
    deadline = time.monotonic() + REQ_BUDGET_SEC

    def _time_left() -> float:
        return float(deadline - time.monotonic())

    def _deadline_exceeded() -> bool:
        return time.monotonic() > deadline

    stable_top5_sec = int(req.stable_top5_sec or env_int("TOP5_STABLE_SEC", TOP5_STABLE_SEC_DEFAULT))
    stable_top5_sec = max(0, min(stable_top5_sec, 120))
    avoid_franchise = bool(req.avoid_franchise) if req.avoid_franchise is not None else bool(env_bool("AVOID_FRANCHISE", "0"))
    balance_types = bool(req.balance_types) if req.balance_types is not None else bool(env_bool("BALANCE_TYPES", "1"))
    req_area = _clean_str(req.area)
    req_must_area = _clean_str(req.must_area)
    req_cafe_type = _clean_str(req.cafe_type)

    if not user_text:
        return ChatResponse(
            ok=False,
            axes={k: 0.0 for k in KEYS},
            wave={
                "n": 0,
                "dominant": "J",
                "dominant_name": "기쁨",
                "trend": "flat",
                "drift": 0.0,
                "pace": "mid",
                "delta": {},
                "loop": {"score": 0.0},
                "topic": "general",
            },
            action_id=0,
            goal="CLARIFY",
            style="NORMAL",
            reply="빈 문장은 처리 못 해. 한 문장만 보내줘.",
            debug={"reason": "empty_text"},
        )

    amp_flag_early = _should_force_amplify(user_text)
    # ============================================================
    # ✅ FAST PATH: 카카오를 먼저 친다 (deadline 소비 전에)
    # - infer_with_retries / PPO / policy 만들기 전에 후보를 확보해야 함
    # ============================================================
    pre_kakao_raw: List[Dict[str, Any]] = []
    pre_kakao_places: List[Dict[str, Any]] = []
    pre_kakao_dbg: Dict[str, Any] = {}
    pre_entity_best = {"place_id": "", "place_name": "", "address": "", "phone": "", "place_url": ""}

    # action_id는 아직 없으니 "amp_flag_early"로만 1차 판단
    if amp_flag_early and (not _deadline_exceeded()):
        # area/type 우선순위: must_area > text_area > req_area
        tmp0 = _detect_area_type_from_text(user_text)
        text_area0 = _clean_str(tmp0.get("area"))
        area0 = (req_must_area or text_area0 or req_area)  # 이미 clean 됨 -> None 가능

        cafe_type0 = (req_cafe_type or _clean_str(tmp0.get("type")) or "카페")
        core_q0 = build_kakao_safe_query(area=area0, cafe_type=cafe_type0)
        fallbacks0 = build_kakao_fallback_queries(area=area0, cafe_type=cafe_type0)

        # ✅ 카카오 먼저 때리고, 시간이 남으면 뒤에서 RAG/정교화
        last_dbg0 = {}
        got = False
        for q_try0 in fallbacks0:
            if _deadline_exceeded():
                last_dbg0 = {"error": "time_budget_exceeded_before_kakao_fastpath"}
                break

            pre_kakao_raw, kd0 = kakao_place_search_multi(
                q_try0,
                size_per_page=15,
                want_total=30,
                area=area0,
            )
            last_dbg0 = dict(kd0 or {})
            if pre_kakao_raw:
                got = True
                break

        pre_kakao_dbg = dict(last_dbg0 or {})

        # blocklist + area filter + top5 pick (fast)
        if pre_kakao_raw:
            pre_kakao_raw, _ = _filter_blocklist(pre_kakao_raw)
            if area0:
                pre_kakao_raw = [p for p in pre_kakao_raw if _area_match(p, area0)]

            seed_bucket0 = int(time.time() // 30)
            pre_kakao_places, _pick_dbg0 = pick_top5_diverse(
                pre_kakao_raw,
                avoid_franchise=avoid_franchise,
                stable_seed_bucket=seed_bucket0,
                balance_types=balance_types,
            )

            if area0:
                pre_kakao_places = [p for p in pre_kakao_places if _area_match(p, area0)]
            pre_kakao_places, _ = _filter_blocklist(pre_kakao_places)

            best0 = pre_kakao_places[0] if pre_kakao_places else None
            if best0:
                pre_entity_best = {
                    "place_id": str(best0.get("id", "") or "").strip(),
                    "place_name": str(best0.get("place_name", "") or "").strip(),
                    "address": str(best0.get("address", "") or "").strip()
                    or (
                        str(best0.get("road_address_name", "") or "").strip()
                        or str(best0.get("address_name", "") or "").strip()
                    ),
                    "phone": str(best0.get("phone", "") or "").strip(),
                    "place_url": str(best0.get("place_url", "") or "").strip(),
                }

    # 1) 8축 추론
    if LITE_MODE or (MODEL is None):
        res = {"ok": True, "axes": {"F":0.2,"A":0.2,"D":0.1,"J":0.2,"C":0.1,"G":0.1,"T":0.2,"R":0.2}, "debug": {"lite": True}}
    else:
        res = infer_with_retries(TOK, MODEL, user_text)

    axes = normalize_axes(res.get("axes", {}))
    ok = bool(res.get("ok", False))

    # --- axes harden: 8축 키를 항상 채워 넣기 ---
    default_axes = {"F": 0.0, "A": 0.0, "D": 0.05, "J": 0.2, "C": 0.15, "G": 0.0, "T": 0.05, "R": 0.85}

    if not isinstance(axes, dict):
        axes = {}

    # 키 누락 보정 (부분 dict도 안전)
    axes = {**default_axes, **axes}

    # LITE_MODE면 더 강하게 보정 (원하면)
    if LITE_MODE and (not ok):
        axes = default_axes


    if not isinstance(axes, dict) or len(axes) == 0:
        axes = {"F": 0.0, "A": 0.0, "D": 0.05, "J": 0.2, "C": 0.15, "G": 0.0, "T": 0.05, "R": 0.85}

    # 2) wave summary
    wave = load_wave_summary_from_jsonl(LOG_PATH, window=window, min_window=3)

    obs14, loop_score, wave_feats = make_obs14(axes=axes, log_path=LOG_PATH)
    if not isinstance(wave.get("loop", {}), dict):
        wave["loop"] = {}
    wave["loop"]["score"] = float(loop_score)
    wave["feats14"] = wave_feats

    # 4) diversity 파라미터
    force_rlmodule = env_bool("PPO_DEBUG_FORCE_RLMODULE", "0")
    sample_from_logits_flag = env_bool("PPO_SAMPLE_FROM_LOGITS", "1")
    eps_greedy = float(np.clip(env_float("PPO_EPS_GREEDY", 0.25), 0.0, 1.0))
    temperature = float(np.clip(env_float("PPO_TEMPERATURE", 1.6), 0.05, 5.0))
    topk = env_int("PPO_TOPK", 5)

    anti_collapse = env_bool("PPO_ANTI_COLLAPSE", "1")
    anti_k = env_int("PPO_ANTI_K", 6)
    anti_eps = float(np.clip(env_float("PPO_ANTI_EPS", 0.60), 0.0, 1.0))

    debug_rl: Dict[str, Any] = {
        "rl_ready": bool(RL_READY),
        "used_ppo": False,
        "error": None,
        "mode": None,
        "obs14_shape": list(obs14.shape),
        "eps_greedy": float(eps_greedy),
        "temperature": float(temperature),
        "sample_from_logits": bool(sample_from_logits_flag),
        "force_rlmodule": bool(force_rlmodule),
        "anti_collapse": bool(anti_collapse),
        "anti_k": int(anti_k),
        "anti_eps": float(anti_eps),
        "amplify_flag_early": bool(amp_flag_early),
        "user_text": user_text,
        "req_budget_sec": float(REQ_BUDGET_SEC),
        "time_left_sec_at_rl": None,
    }

    # 5) PPO action 선택
    action_id = 0
    if RL_READY and RL_ALGO is not None and (not _deadline_exceeded()):
        debug_rl["time_left_sec_at_rl"] = float(_time_left())
        try:
            if np.random.rand() < eps_greedy:
                action_id = int(np.random.randint(0, int(NUM_ACTIONS)))
                debug_rl["used_ppo"] = True
                debug_rl["mode"] = "eps_greedy_random"
            else:
                logits = None

                if force_rlmodule or sample_from_logits_flag:
                    try:
                        m = RL_ALGO.get_module()
                        batch_np = {"obs": np.expand_dims(obs14, axis=0).astype(np.float32, copy=False)}
                        out = m.forward_inference(batch_np)

                        if isinstance(out, dict) and "action_dist_inputs" in out:
                            log = np.array(out["action_dist_inputs"])
                            logits = log[0] if log.ndim == 2 else log.reshape(-1)
                            debug_rl["mode"] = "rlmodule_logits"
                            debug_rl["used_ppo"] = True
                        else:
                            raise KeyError("forward_inference missing action_dist_inputs")
                    except Exception as e_mod:
                        debug_rl["rlmodule_error"] = repr(e_mod)
                        logits = None

                if logits is not None:
                    debug_rl["top5_logits"] = topk_pairs(logits, k=topk)

                    if sample_from_logits_flag:
                        action_id, probs = sample_from_logits(logits, temperature=temperature)
                        debug_rl["mode"] = str(debug_rl["mode"]) + "+sample"
                        debug_rl["top5_probs"] = topk_pairs(probs, k=topk)
                    else:
                        action_id = int(np.argmax(logits))
                        debug_rl["mode"] = str(debug_rl["mode"]) + "+argmax"
                else:
                    out = RL_ALGO.compute_single_action(obs14, explore=True)
                    action_id = int(out) if not isinstance(out, (tuple, list)) else int(out[0])
                    debug_rl["used_ppo"] = True
                    debug_rl["mode"] = "compute_single_action(explore=True)"

            if action_id < 0 or action_id >= int(NUM_ACTIONS):
                action_id = 0

        except Exception as e:
            debug_rl["error"] = repr(e)
            action_id = 0

    # 6) anti-collapse
    try:
        LAST_ACTIONS.append(int(action_id))
        if anti_collapse and len(LAST_ACTIONS) >= int(anti_k):
            tail = list(LAST_ACTIONS)[-int(anti_k):]
            if len(set(tail)) == 1 and np.random.rand() < anti_eps:
                old = int(action_id)
                candidates = [i for i in range(int(NUM_ACTIONS)) if i != old]
                action_id = int(np.random.choice(candidates))
                debug_rl["mode"] = str(debug_rl.get("mode")) + "+anti_collapse"
                debug_rl["anti_from"] = old
                debug_rl["anti_to"] = int(action_id)
    except Exception:
        pass

    # 7) action -> (goal, style)
    _g, _s = action_pair(int(action_id))

    amp_flag = _should_force_amplify(user_text)
    debug_rl["amplify_flag"] = bool(amp_flag)
    debug_rl["amplify_hits"] = [k for k in AMPLIFY_KEYWORDS if k.lower() in (user_text or "").lower()][:10]
    debug_rl["action_pair"] = {"goal": _g, "style": _s}
    debug_rl["action_id"] = int(action_id)

    # ✅ RL이 꺼져도 AMPLIFY로 들어가게 강제
    try:
        if amp_flag and int(action_id) not in AMPLIFY_ACTION_IDS:
            old = int(action_id)
            action_id = 9
            _g, _s = action_pair(int(action_id))
            debug_rl["mode"] = str(debug_rl.get("mode")) + "+force_amplify"
            debug_rl["force_amplify"] = {"from": old, "to": int(action_id), "by": "keywords"}
            debug_rl["action_pair"] = {"goal": _g, "style": _s}
            debug_rl["action_id"] = int(action_id)
    except Exception:
        pass

    # ============================================================
    # ✅ AMPLIFY: 카카오 후보 TOP5 (+ RAG 옵션은 유지)
    # ============================================================
    kakao_places: List[Dict[str, Any]] = []
    kakao_raw: List[Dict[str, Any]] = []
    kakao_dbg: Dict[str, Any] = {}
    entity_best = {"place_id": "", "place_name": "", "address": "", "phone": "", "place_url": ""}

    rag_on = (str(_g).upper() == "AMPLIFY") and env_bool("RAG_ON", "1")
    base_dir = os.path.dirname(__file__)
    rag_store_dir = os.path.join(base_dir, "rag", "store")
    want_store = "amplify" if str(_g).upper() == "AMPLIFY" else None
    _store_dir = os.path.join(rag_store_dir, want_store) if want_store else rag_store_dir

    rag_index_path = os.path.join(_store_dir, "index.faiss")
    rag_meta_path = os.path.join(_store_dir, "meta.jsonl")
    rag_index_ready = bool(os.path.exists(rag_index_path) and os.path.exists(rag_meta_path))

    _tmp_filters = _detect_area_type_from_text(user_text) if rag_on else {}
    rag_filters = _tmp_filters if ("area" in _tmp_filters and "type" in _tmp_filters) else {}

    rag_each_show = int(np.clip(env_int("RAG_EACH_SHOW", 3), 1, 5))
    rag_each_k = int(np.clip(env_int("RAG_EACH_K", 8), 3, 12))

    kakao_candidate_pool = 30
    kakao_size_per_page = 15

    hint_weights = _detect_hint_weights(user_text)
    hint_alpha = float(np.clip(env_float("RAG_HINT_ALPHA", 0.35), 0.0, 1.0))

    debug_rag: Dict[str, Any] = {
        "rag_on": bool(rag_on),
        "index_ready": bool(rag_index_ready),
        "store_dir": _store_dir,
        "store_name": want_store or "",
        "index_path": rag_index_path,
        "meta_path": rag_meta_path,
        "used": False,
        "k_each": int(rag_each_k),
        "show_each": int(rag_each_show),
        "error": None,
        "filters_in": dict(_tmp_filters or {}),
        "filters_gate": dict(rag_filters or {}),
        "by_place": [],
        "kakao_query": "",
        "kakao_area": "",
        "kakao_type": "",
        "kakao_raw_size": int(kakao_candidate_pool),
        "kakao_debug": {},
        "rag_search_backend": "rag.search.search",
        "top5_stable_sec": int(stable_top5_sec),
        "avoid_franchise": bool(avoid_franchise),
        "balance_types": bool(balance_types),
        "cache_hit": False,
        "cache_age_sec": None,
        "diverse_seed_bucket": None,
        "diverse_pick_debug": {},
        "hint_weights": dict(hint_weights),
        "hint_alpha": float(hint_alpha),
        "hint_rerank_applied": False,
        "hint_rerank_debug": {},
        "hint_rerank_skipped": None,
        "blocklist_dropped_raw": 0,
        "blocklist_dropped_top5": 0,
        "req_budget_sec": float(REQ_BUDGET_SEC),
        "time_left_sec_at_rag": None,
    }

    rag_hits_all: List[Dict[str, Any]] = []
    # ✅ NEW: place별 RAG hits 재사용(중복 호출 제거용)
    rag_hits_by_place_id: Dict[str, List[Dict[str, Any]]] = {}

    reply, _ignored_goal, _ignored_style = policy_v2_friend_with_action(
        user_text=user_text,
        axes=axes,
        wave_summary=wave,
        action_id=int(action_id),
        stable_sec=int(stable_top5_sec),
    )

# ✅ goal/style은 action_pair 기준으로 고정
    goal, style = str(_g), str(_s)

    

    # ============================================================
    # ✅ 멀티 후보 블록 생성
    # ============================================================
    try:
        if str(_g).upper() == "AMPLIFY":

            # ✅ 우선순위: must_area > area > 텍스트 추출 (PATCH 4-2)
            _tmp = _detect_area_type_from_text(user_text)

            text_area = _clean_str(_tmp.get("area"))
            area = (req_must_area or text_area or req_area)  # None 가능

            cafe_type = (req_cafe_type or _clean_str(_tmp.get("type")) or "카페")
            core_q = build_kakao_safe_query(area=area, cafe_type=cafe_type)
            fallbacks = build_kakao_fallback_queries(area=area, cafe_type=cafe_type)

            debug_rag["kakao_query"] = core_q
            debug_rag["kakao_area"] = area
            debug_rag["kakao_type"] = cafe_type

            cache_key = _normalize_cache_key(
                f"{core_q}||area={area}||type={cafe_type}||avoid_franchise={avoid_franchise}||balance_types={balance_types}"
            )
            now = time.time()

            cache_rec = _TOP5_CACHE.get(cache_key)
            if cache_rec and stable_top5_sec > 0:
                age = float(now - float(cache_rec.get("ts", 0.0)))
                if age <= float(stable_top5_sec):
                    kakao_raw = list(cache_rec.get("kakao_raw", []) or [])
                    kakao_dbg = dict(cache_rec.get("kakao_dbg", {}) or {})
                    kakao_places = list(cache_rec.get("kakao_places", []) or [])

                    debug_rag["cache_hit"] = True
                    debug_rag["cache_age_sec"] = float(age)
                    debug_rag["diverse_seed_bucket"] = cache_rec.get("seed_bucket")
                    debug_rag["diverse_pick_debug"] = dict(cache_rec.get("pick_dbg", {}) or {})
                    debug_rag["diverse_pick_debug"]["balance_types"] = bool(balance_types)
                    if isinstance(cache_rec.get("hint_rerank_debug"), dict):
                        debug_rag["hint_rerank_debug"] = dict(cache_rec.get("hint_rerank_debug") or {})
                        debug_rag["hint_rerank_applied"] = bool(cache_rec.get("hint_rerank_applied", False))
                else:
                    _TOP5_CACHE.pop(cache_key, None)

            if not debug_rag.get("cache_hit"):

                # ✅ 1) FAST PATH 결과가 있으면: 카카오 재호출 금지, 그대로 재사용
                if pre_kakao_raw:
                    # ✅ FAST PATH 결과가 있으면 재사용 (카카오 재호출 금지)
                    kakao_raw = list(pre_kakao_raw)
                    kakao_places = list(pre_kakao_places)
                    kakao_dbg = dict(pre_kakao_dbg or {})
                    entity_best = dict(pre_entity_best or {})
                    debug_rag["kakao_debug"] = dict(kakao_dbg or {})

                    # ✅ 표시용 kakao_query도 실제 used_query로 보정(로그 깔끔)
                    debug_rag["kakao_query"] = debug_rag.get("kakao_query") or kakao_dbg.get("used_query", "")

                # ✅ 2) FAST PATH가 없으면: 기존대로 카카오 fallback 루프 실행
                else:
                    last_dbg = {}
                    for q_try in fallbacks:
                        if _deadline_exceeded():
                            debug_rag["error"] = "time_budget_exceeded_before_kakao"
                            break

                        kakao_raw, kakao_dbg = kakao_place_search_multi(
                            q_try,
                            size_per_page=kakao_size_per_page,
                            want_total=kakao_candidate_pool,
                            area=area,
                        )
                        last_dbg = dict(kakao_dbg or {})
                        if kakao_raw:
                            debug_rag["kakao_query"] = q_try
                            break

                    debug_rag["kakao_debug"] = dict(last_dbg or {})
                    kakao_dbg = dict(last_dbg or {})

                # ============================================================
                # ✅ 여기부터는 “공통 파이프라인”
                # ============================================================

                seed_bucket = int(time.time() // 30)
                pick_dbg = {}
                kakao_raw, drop_raw = _filter_blocklist(kakao_raw)
                debug_rag["blocklist_dropped_raw"] = int(drop_raw)

                area_drop = 0
                if area:
                    before = len(kakao_raw or [])
                    # 🔥 HOTFIX: area filter temporarily disabled
                    # kakao_raw = [p for p in (kakao_raw or []) if _area_match(p, area)]
                    area_drop = before - len(kakao_raw or [])
                debug_rag["area_filter"] = {"area": area, "dropped_raw": int(area_drop)}
                if not kakao_places:
                    kakao_places, pick_dbg = pick_top5_diverse(
                    kakao_raw,
                    avoid_franchise=avoid_franchise,
                    stable_seed_bucket=seed_bucket,
                    balance_types=balance_types,
                    )
                else:
                    # fastpath에서 이미 pick된 top5를 재사용한 경우
                    pick_dbg = {"reused_pre_kakao_places": True}

                if area:
                    debug_rag.setdefault("area_filter", {"area": area, "dropped_raw": 0})
                    before5 = len(kakao_places or [])
                    # 🔥 HOTFIX: area filter temporarily disabled
                    # kakao_places = [p for p in (kakao_places or []) if _area_match(p, area)]
                    debug_rag["area_filter"]["dropped_top5"] = int(before5 - len(kakao_places or []))
                kakao_places, drop_top = _filter_blocklist(kakao_places)
                debug_rag["blocklist_dropped_top5"] = int(drop_top)

                debug_rag["diverse_pick_debug"] = dict(pick_dbg or {})
                debug_rag["diverse_pick_debug"]["balance_types"] = bool(balance_types)

                # ✅ RAG 힌트 스코어링으로 TOP5 재정렬 (시간 충분할 때만)
                hint_scores_by_id: Dict[str, float] = {}
                hint_dbg_by_id: Dict[str, Any] = {}

                debug_rag["time_left_sec_at_rag"] = float(_time_left())

                allow_hint_rerank = (
                    rag_on
                    and rag_index_ready
                    and sum(float(hint_weights.get(k, 0.0)) for k in _HINT_KEYS) > 0.0
                    and (_time_left() > 3.0)
                )

                if allow_hint_rerank:
                    for p in (kakao_places or [])[:5]:
                        if _deadline_exceeded() or (_time_left() <= 1.2):
                            debug_rag["hint_rerank_skipped"] = "time_budget_low_during_rerank"
                            break

                        pname = str(p.get("place_name", "") or "").strip()
                        pid = str(p.get("id", "") or "").strip()
                        if not pname or not pid:
                            continue

                        try:
                            area_f = str(rag_filters.get("area", "") or "")
                            ptype_f = str(rag_filters.get("type", "") or "")

                            hint_terms = []
                            if hint_weights.get("quiet", 0.0) > 0:
                                hint_terms.append("조용")
                            if hint_weights.get("outlet", 0.0) > 0:
                                hint_terms.append("콘센트")
                            if hint_weights.get("table", 0.0) > 0:
                                hint_terms.append("테이블")
                            q_for_rag = (pname + " " + " ".join(hint_terms)).strip()

                            out = rag_search(
                                query=q_for_rag,
                                store_dir=_store_dir,
                                top_k=int(rag_each_k),
                                area=area_f,
                                ptype=ptype_f,
                                # ✅ FIX: 과한 탐색 완화 (*10 → *3)
                                k_each=max(30, int(rag_each_k) * 3),
                                raw_fallback=True,
                            )
                            hits = out.get("results", []) if isinstance(out, dict) else []
                            if not isinstance(hits, list):
                                hits = []
                            hits2 = hits[: max(1, int(rag_each_show))]

                            # ✅ NEW: hits 저장(출력에서 재사용)
                            rag_hits_by_place_id[pid] = list(hits2)

                            hs, hs_dbg = score_hints_from_rag_hits(hits2, hint_weights=hint_weights)
                            hint_scores_by_id[pid] = float(hs)
                            hint_dbg_by_id[pid] = hs_dbg

                            rag_hits_all.extend(hits2)

                            debug_rag["by_place"].append(
                                {
                                    "place_id": pid,
                                    "place_name": pname,
                                    "query_for_rag": q_for_rag,
                                    "n_hits": int(len(hits2)),
                                    "top_urls": [str(h.get("url", "")) for h in hits2[:3]],
                                    "filters_used": {"area": area_f, "ptype": ptype_f},
                                    "hint_score": float(hs),
                                    "hint_debug": hs_dbg,
                                }
                            )
                            debug_rag["used"] = True
                        except Exception as e_r:
                            debug_rag["by_place"].append({"place_id": pid, "place_name": pname, "error": repr(e_r)})

                    # ✅ 힌트 리랭킹 적용(점수 있을 때)
                    if hint_scores_by_id:
                        before_names = [str(p.get("place_name", "") or "") for p in (kakao_places or [])[:5]]
                        kakao_places, rr_dbg = rerank_top5_by_hints(
                            kakao_places[:5],
                            hint_scores_by_id,
                            alpha=hint_alpha,
                        )
                        after_names = [str(p.get("place_name", "") or "") for p in (kakao_places or [])[:5]]
                        debug_rag["hint_rerank_applied"] = True
                        rr_dbg["before"] = before_names
                        rr_dbg["after"] = after_names
                        rr_dbg["hint_dbg_by_id"] = hint_dbg_by_id
                        debug_rag["hint_rerank_debug"] = rr_dbg
                    else:
                        debug_rag["hint_rerank_skipped"] = debug_rag.get("hint_rerank_skipped") or "no_hint_scores"
                else:
                    debug_rag["hint_rerank_applied"] = False
                    debug_rag["hint_rerank_skipped"] = "disabled_or_index_not_ready_or_no_hints_or_low_time"

                kakao_raw, _ = _filter_blocklist(kakao_raw)
                kakao_places, _ = _filter_blocklist(kakao_places)

                _cache_prune_if_needed()
                _TOP5_CACHE[cache_key] = {
                    "ts": float(now),
                    "kakao_raw": list(kakao_raw or []),
                    "kakao_dbg": dict(kakao_dbg or {}),
                    "kakao_places": list(kakao_places or []),
                    "seed_bucket": int(seed_bucket),
                    "pick_dbg": dict(pick_dbg or {}),
                    "hint_rerank_applied": bool(debug_rag.get("hint_rerank_applied")),
                    "hint_rerank_debug": dict(debug_rag.get("hint_rerank_debug") or {}),
                }

            place_best = kakao_places[0] if kakao_places else None
            if place_best:
                entity_best = {
                    "place_id": str(place_best.get("id", "") or "").strip(),
                    "place_name": str(place_best.get("place_name", "") or "").strip(),
                    "address": str(place_best.get("address", "") or "").strip()
                    or (
                        str(place_best.get("road_address_name", "") or "").strip()
                        or str(place_best.get("address_name", "") or "").strip()
                    ),
                    "phone": str(place_best.get("phone", "") or "").strip(),
                    "place_url": str(place_best.get("place_url", "") or "").strip(),
                }

            blocks: List[str] = []
            blocks.append("📍 후보 장소 TOP5 (카카오맵: 주소/전화/URL)")
            blocks.append(
                f"- 옵션: stable_top5_sec={stable_top5_sec} avoid_franchise={avoid_franchise} balance_types={balance_types} "
                f"cache_hit={debug_rag.get('cache_hit')} age={debug_rag.get('cache_age_sec')}"
            )
            blocks.append(f"- SAFE_QUERY: kakao_query='{debug_rag.get('kakao_query')}' area='{area}' type='{cafe_type}'")
            blocks.append(
                f"- 블랙리스트: dropped_raw={debug_rag.get('blocklist_dropped_raw')} dropped_top5={debug_rag.get('blocklist_dropped_top5')}"
            )
            blocks.append(
                f"- RAG 힌트리랭킹: applied={debug_rag.get('hint_rerank_applied')} "
                f"weights={debug_rag.get('hint_weights')} alpha={debug_rag.get('hint_alpha')} "
                f"skipped={debug_rag.get('hint_rerank_skipped')}"
            )
            blocks.append(f"- TIME_BUDGET: budget={REQ_BUDGET_SEC}s time_left={_time_left():.2f}s")

            if not kakao_raw:
                stt = (kakao_dbg or {}).get("status")
                er = (kakao_dbg or {}).get("error")
                hk = (kakao_dbg or {}).get("has_key")
                kp = (kakao_dbg or {}).get("key_prefix")
                wf = (kakao_dbg or {}).get("want_cafe_only")
                fo = (kakao_dbg or {}).get("filtered_out")
                fb = (kakao_dbg or {}).get("filtered_blocklist")
                pages = (kakao_dbg or {}).get("pages")
                blocks.append(
                    f"- (카카오 후보 없음) status={stt} error={er} has_key={hk} key={kp} "
                    f"want_cafe_only={wf} filtered_out={fo} filtered_blocklist={fb} pages={pages}"
                )
            else:
                for idx, p in enumerate(kakao_places[:5], start=1):
                    pname = str(p.get("place_name", "") or "").strip()
                    paddr = str(p.get("address", "") or "").strip() or (
                        str(p.get("road_address_name", "") or "").strip()
                        or str(p.get("address_name", "") or "").strip()
                    )
                    pphone = str(p.get("phone", "") or "").strip()
                    purl = str(p.get("place_url", "") or "").strip()

                    blocks.append(f"\n[{idx}] {pname}" + (f" · {paddr}" if paddr else ""))
                    if pphone:
                        blocks.append(f"  - phone: {pphone}")
                    if purl:
                        blocks.append(f"  - url: {purl}")

                    # ✅ FIX: 여기서 rag_search 재호출 금지 (힌트리랭킹에서 저장한 hits 재사용)
                    if rag_on and rag_index_ready:
                        pid = str(p.get("id", "") or "").strip()
                        hits = rag_hits_by_place_id.get(pid, [])
                        if hits:
                            rag_hits_all.extend(hits)

            reply = f"{reply}\n\n" + "\n".join(blocks).strip()

    except Exception as e:
        debug_rag["error"] = repr(e)

    wave["rag"] = [{"url": h.get("url", ""), "score": float(h.get("score", 0.0) or 0.0)} for h in (rag_hits_all or [])]

    wave_for_log = dict(wave) if isinstance(wave, dict) else {}
    wave_for_log["action"] = {"id": int(action_id), "goal": str(goal), "style": str(style)}
    if not isinstance(wave_for_log.get("loop", {}), dict):
        wave_for_log["loop"] = {}
    wave_for_log["loop"].setdefault("score", float(loop_score))

    # ✅ FIX: LITE_MODE=1이면 append_log가 None일 수 있으니 callable 가드
    if callable(append_log):
        try:
            append_log(LOG_PATH, user_text, axes, ok, reply=reply, wave_state=wave_for_log)
        except TypeError:
            # 구버전 시그니처 호환(혹시 positional-only였던 경우)
            try:
                append_log(LOG_PATH, user_text, axes, ok, reply, wave_for_log)
            except Exception:
                try:
                    append_log(LOG_PATH, user_text, axes, ok)
                except Exception:
                    pass
        except Exception:
            pass


    base_debug = res.get("debug", {})
    if not isinstance(base_debug, dict):
        base_debug = {}

    debug_out = dict(base_debug)
    debug_out["rl"] = debug_rl
    debug_out["rag"] = debug_rag

    debug_out["naver_cafe"] = {"enabled": False}

    debug_out["kakao_place"] = {
        "enabled": bool(KAKAO_REST_API_KEY),
        "raw_count": int(len(kakao_raw)),
        "kept_count": int(len(kakao_places)),
        "use_bias": bool(env_bool("KAKAO_USE_BIAS", "0")),
        "kakao_debug": dict(kakao_dbg or {}),
        "candidate_pool_target": int(30),
        "candidates": [
            {
                "id": str(p.get("id", "") or ""),
                "name": str(p.get("place_name", "") or ""),
                "addr": str(p.get("address", "") or "").strip()
                or (str(p.get("road_address_name", "") or "") or str(p.get("address_name", "") or "")),
                "phone": str(p.get("phone", "") or ""),
                "url": str(p.get("place_url", "") or ""),
                "category": str(p.get("category_name", "") or ""),
            }
            for p in (kakao_places or [])[:5]
        ],
        "selected_top1": entity_best if entity_best.get("place_name") else {},
        "options": {
            "avoid_franchise": bool(avoid_franchise),
            "stable_top5_sec": int(stable_top5_sec),
            "balance_types": bool(balance_types),
        },
        "blocklist": [rx.pattern for rx in BLOCKLIST_PATTERNS],
        "kakao_max_pages": int(env_int("KAKAO_MAX_PAGES", 2)),
    }
    debug_out["entity"] = entity_best

    return ChatResponse(
        ok=ok,
        axes=axes,
        wave=wave,
        action_id=int(action_id),
        goal=str(goal),
        style=str(style),
        reply=str(reply),
        debug=debug_out,
    )


# ============================================================
# ✅ 아래 함수들은 원본 유지
# ============================================================
def load_wave_summary_from_jsonl(path: str, window: int = 10, min_window: int = 3) -> Dict[str, Any]:
    base = {
        "n": 0,
        "dominant": "J",
        "dominant_name": "기쁨",
        "trend": "flat",
        "drift": 0.0,
        "pace": "mid",
        "delta": {},
        "loop": {},
        "topic": "general",
    }
    if not os.path.exists(path):
        return base

    axes_list: List[Dict[str, float]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max(1, int(window)):]
    except Exception:
        return base

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        labels = rec.get("labels") or rec.get("axes") or {}
        if not isinstance(labels, dict):
            continue

        ax = {k: clamp01(labels.get(k, 0.0)) for k in KEYS}
        axes_list.append(ax)

    n = len(axes_list)
    if n < int(min_window):
        out = dict(base)
        out["n"] = n
        return out

    last = axes_list[-1]
    first = axes_list[0]

    dom_axis = max(["F", "A", "D", "J"], key=lambda k: float(last.get(k, 0.0)))
    dom_name = {"F": "불안", "A": "분노", "D": "우울", "J": "기쁨"}[dom_axis]
    delta = {k: float(last[k] - first[k]) for k in KEYS}

    drifts = []
    for i in range(1, n):
        drifts.append(sum(abs(axes_list[i][k] - axes_list[i - 1][k]) for k in KEYS) / len(KEYS))
    drift = float(sum(drifts) / max(1, len(drifts)))

    T = float(last.get("T", 0.0))
    R = float(last.get("R", 0.0))
    if T >= 0.65:
        pace = "high"
    elif R >= 0.65:
        pace = "low"
    else:
        pace = "mid"

    dom_d = float(delta.get(dom_axis, 0.0))
    if dom_d > 0.05:
        trend = "up"
    elif dom_d < -0.05:
        trend = "down"
    else:
        trend = "flat"

    return {
        "n": n,
        "dominant": dom_axis,
        "dominant_name": dom_name,
        "trend": trend,
        "drift": drift,
        "pace": pace,
        "delta": delta,
        "loop": {},
        "topic": "general",
    }


def axes_to_8vec(axes: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(axes[k]) for k in KEYS], dtype=np.float32)


def _signed_to_01(x: float) -> float:
    return clamp01((float(x) + 1.0) * 0.5)


def calc_drift(prev8: np.ndarray, cur8: np.ndarray) -> float:
    return float(np.mean(np.abs(cur8 - prev8)))


def read_last_axes8_from_log(log_path: str, n: int = 8) -> List[np.ndarray]:
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max(1, n):]
    except Exception:
        return []

    out: List[np.ndarray] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        labels = rec.get("labels") or rec.get("axes") or {}
        if not isinstance(labels, dict):
            continue
        ax = {k: clamp01(labels.get(k, 0.0)) for k in KEYS}
        out.append(axes_to_8vec(ax))
    return out


def calc_loop_score_recent(recent_axes8: List[np.ndarray], k: int = 6) -> float:
    if len(recent_axes8) < k:
        return 0.0
    tail = recent_axes8[-k:]
    doms = [KEYS[int(np.argmax(v))] for v in tail]
    most = max(set(doms), key=doms.count)
    ratio = doms.count(most) / float(k)

    drifts = []
    for i in range(1, k):
        drifts.append(float(np.mean(np.abs(tail[i] - tail[i - 1]))))
    drift_mean = float(np.mean(drifts)) if drifts else 0.0

    low_drift_factor = max(0.0, 0.25 - drift_mean) / 0.25
    score = ratio * (0.5 + 0.5 * low_drift_factor)
    return float(np.clip(score, 0.0, 1.0))


def dom_value_and_delta(prev8: np.ndarray, cur8: np.ndarray) -> Tuple[str, float, float]:
    dom_idx = int(np.argmax(cur8))
    dom_axis = KEYS[dom_idx]
    dom_value = float(cur8[dom_idx])
    dom_delta = float(cur8[dom_idx] - prev8[dom_idx])
    return dom_axis, dom_value, dom_delta


def trend_sign_from_dom_delta(dom_delta: float, th: float = 0.05) -> float:
    if dom_delta > th:
        return 1.0
    if dom_delta < -th:
        return -1.0
    return 0.0


def make_obs14(axes: Dict[str, float], log_path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    cur8 = axes_to_8vec(axes)

    recent8 = read_last_axes8_from_log(log_path, n=8)
    prev8 = recent8[-2] if len(recent8) >= 2 else cur8

    drift = float(np.clip(calc_drift(prev8, cur8), 0.0, 1.0))

    recent_for_loop = list(recent8)
    if (not recent_for_loop) or (not np.allclose(recent_for_loop[-1], cur8)):
        recent_for_loop.append(cur8)
    loop_score = float(np.clip(calc_loop_score_recent(recent_for_loop, k=6), 0.0, 1.0))

    dom_axis, dom_value, dom_delta = dom_value_and_delta(prev8, cur8)
    trend_sign = trend_sign_from_dom_delta(dom_delta, th=0.05)
    rt_gap = float(cur8[7] - cur8[6])  # R - T

    obs = np.concatenate(
        [
            cur8,
            np.array([drift], np.float32),
            np.array([loop_score], np.float32),
            np.array([dom_value], np.float32),
            np.array([_signed_to_01(dom_delta)], np.float32),
            np.array([_signed_to_01(trend_sign)], np.float32),
            np.array([_signed_to_01(rt_gap)], np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    feats = {
        "dom_axis": dom_axis,
        "dom_value": float(dom_value),
        "dom_delta": float(dom_delta),
        "trend_sign": float(trend_sign),
        "rt_gap": float(rt_gap),
        "drift": float(drift),
        "loop_score": float(loop_score),
    }
    return obs.reshape((14,)), loop_score, feats


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(x) / float(len(x))


def sample_from_logits(logits: np.ndarray, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
    t = max(0.05, float(temperature))
    z = logits.astype(np.float32) / t
    probs = softmax(z)
    a = int(np.random.choice(len(probs), p=probs))
    return a, probs


def topk_pairs(vals: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    k = max(1, min(int(k), int(vals.shape[-1])))
    idx = np.argsort(vals)[::-1][:k]
    out: List[Dict[str, Any]] = []
    for i in idx:
        g, s = action_pair(int(i))
        out.append({"action_id": int(i), "value": float(vals[i]), "goal": g, "style": s})
    return out


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
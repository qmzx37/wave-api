from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


# -----------------------
# helpers
# -----------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _load_store(store_dir: str) -> Dict[str, str]:
    store_dir = str(store_dir)
    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.jsonl")
    config_path = os.path.join(store_dir, "config.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.faiss not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    return {"index": index_path, "meta": meta_path, "config": config_path}


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# -----------------------
# 부산 지역 alias (넓게)
# - "광안리" 같은 동네명 -> 구/군으로 매핑 + 동네명도 함께 허용
# -----------------------
AREA_ALIASES: Dict[str, List[str]] = {
    "수영구": ["광안리", "광안", "민락", "수영", "남천", "금련산"],
    "해운대구": ["해운대", "센텀", "우동", "중동", "장산", "재송", "송정"],
    "부산진구": ["서면", "전포", "부전", "범천", "양정"],
    "동래구": ["동래", "온천장", "사직", "명륜"],
    "남구": ["대연", "용호", "문현", "경성대", "부경대"],
    "중구": ["남포", "광복", "자갈치"],
    "영도구": ["영도", "태종대"],
    "동구": ["초량", "부산역", "범일"],
    "사하구": ["하단", "다대포", "감천"],
    "강서구": ["명지", "대저"],
    "금정구": ["장전", "부산대", "구서"],
    "연제구": ["연산", "연산동"],
    "사상구": ["사상", "괘법", "덕포"],
    "서구": ["송도", "암남", "송도해수욕장"],
    "북구": ["화명", "덕천", "구포"],
    "기장군": ["기장", "정관", "일광"],
}

def _area_tokens(area: str) -> List[str]:
    a = (area or "").strip()
    if not a:
        return []
    # 사용자가 "광안리"를 넣어도 -> "수영구"+"광안리" 같이 넓게 필터
    tokens = [a]
    # a가 별칭이면 해당 구/군을 같이 추가
    for gugun, aliases in AREA_ALIASES.items():
        if a == gugun:
            tokens.append(gugun)
            tokens.extend(aliases)
            break
        if a in aliases:
            tokens.append(gugun)
            tokens.extend(aliases)
            break
    # 중복 제거
    uniq = []
    for t in tokens:
        if t and t not in uniq:
            uniq.append(t)
    return uniq


# -----------------------
# type tokens (대충 넓게)
# -----------------------
TYPE_TOKENS: Dict[str, List[str]] = {
    "디저트": ["디저트", "케이크", "타르트", "마카롱", "휘낭시에", "쿠키", "푸딩", "젤라또", "아이스크림"],
    "베이커리": ["베이커리", "빵", "빵집", "제과", "브레드", "크루아상"],
    "카페": ["카페", "커피", "라떼", "아메리카노", "티", "찻집"],
    "대형카페": ["대형", "루프탑", "오션뷰", "바다뷰", "뷰", "정원", "테라스", "통창"],
}

def _ptype_tokens(ptype: str) -> List[str]:
    t = (ptype or "").strip()
    if not t:
        return []
    if t in TYPE_TOKENS:
        return TYPE_TOKENS[t]
    # 혹시 사용자가 그냥 "케이크" 같은 걸 넣으면 그 자체도 토큰으로
    return [t]


def _passes_filters(m: Dict[str, Any], area: str, ptype: str) -> bool:
    text = _norm(str(m.get("text", ""))) + " " + _norm(str(m.get("title", ""))) + " " + _norm(str(m.get("shop", "")))

    # area filter
    atoks = _area_tokens(area)
    if atoks:
        # 하나라도 걸리면 통과 (넓게)
        if not any(_norm(x) in text for x in atoks):
            return False

    # type filter
    ttoks = _ptype_tokens(ptype)
    if ttoks:
        if not any(_norm(x) in text for x in ttoks):
            return False

    return True


# -----------------------
# main search
# -----------------------
def search(
    query: str,
    store_dir: str,
    top_k: int = 5,
    model_name: Optional[str] = None,
    must_include_any: Optional[List[str]] = None,
    must_exclude_any: Optional[List[str]] = None,
    min_score: float = 0.0,
    # ✅ NEW
    area: str = "",
    ptype: str = "",
    k_each: int = 30,
    raw_fallback: bool = True,
) -> Dict[str, Any]:
    """
    FAISS(IP=cosine) + SentenceTransformer로 검색.
    - 먼저 넓게(k_each) 찾고 → area/ptype로 필터링 → top_k 반환
    - 필터 결과가 0개면 raw_fallback=True일 때 필터 없는 결과로 fallback
    """
    q = (query or "").strip()
    if not q:
        return {"ok": False, "error": "empty_query"}

    if faiss is None:
        return {"ok": False, "error": "faiss not installed/import failed"}
    if SentenceTransformer is None:
        return {"ok": False, "error": "sentence-transformers import failed"}

    paths = _load_store(store_dir)

    cfg = {}
    with open(paths["config"], "r", encoding="utf-8") as f:
        cfg = json.load(f)

    use_model = model_name or cfg.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"

    # 1) load meta
    meta_lines = _read_jsonl(paths["meta"])
    if not meta_lines:
        return {"ok": False, "error": "empty_meta"}

    # 2) load index
    index = faiss.read_index(paths["index"])

    # 3) embed query
    model = SentenceTransformer(use_model)
    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    # 4) search wide
    top_k = max(1, int(top_k))
    k_each = max(top_k, int(k_each))

    D, I = index.search(q_emb, k_each)
    idxs = I[0].tolist()
    sims = D[0].tolist()

    # 5) raw results
    raw_results: List[Dict[str, Any]] = []
    for rank, (mi, sim) in enumerate(zip(idxs, sims), start=1):
        if mi < 0 or mi >= len(meta_lines):
            continue
        if float(sim) < float(min_score):
            continue
        m = meta_lines[mi]

        # include/exclude quick gate
        joined = _norm(str(m.get("title","")) + " " + str(m.get("text","")) + " " + str(m.get("shop","")))
        if must_include_any:
            if not any(_norm(x) in joined for x in must_include_any if (x or "").strip()):
                continue
        if must_exclude_any:
            if any(_norm(x) in joined for x in must_exclude_any if (x or "").strip()):
                continue

        raw_results.append(
            {
                "rank": rank,
                "score": float(sim),
                "chunk_id": m.get("chunk_id", ""),
                "doc_id": m.get("doc_id", ""),
                "chunk_idx": m.get("chunk_idx", 0),
                "title": m.get("title", ""),
                "url": m.get("url", ""),
                "source": m.get("source", ""),
                "fetched_at": m.get("fetched_at", ""),
                "shop": m.get("shop", ""),
                "text": m.get("text", ""),
            }
        )

    # 6) filter by area/ptype
    filtered: List[Dict[str, Any]] = []
    if area or ptype:
        for r in raw_results:
            # r에는 meta 텍스트가 이미 들어있어서 여기로 필터링 가능
            if _passes_filters(r, area=area, ptype=ptype):
                filtered.append(r)

    final_results = filtered[:top_k] if (area or ptype) else raw_results[:top_k]

    # 7) fallback
    if (area or ptype) and (len(final_results) == 0) and raw_fallback:
        final_results = raw_results[:top_k]

    return {
        "ok": True,
        "query": q,
        "top_k": top_k,
        "k_each": k_each,
        "store_dir": store_dir,
        "model_name": use_model,
        "num_meta": len(meta_lines),
        "filters": {"area": area, "ptype": ptype, "raw_fallback": raw_fallback, "min_score": float(min_score)},
        "results": final_results,
        "debug_counts": {"raw": len(raw_results), "filtered": len(filtered)},
    }

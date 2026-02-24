# rag/retrieve.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


BASE_DIR = os.path.dirname(__file__)
DEFAULT_STORE_DIR = os.path.join(BASE_DIR, "store")

_MODEL_CACHE: Dict[str, Any] = {}
_INDEX_CACHE: Dict[str, Any] = {}
_META_CACHE: Dict[str, Any] = {}

DENY_SOURCES_DEFAULT = {"naver_news"}


def _is_blocked(meta: dict, deny_sources=None) -> bool:
    deny_sources = set(deny_sources or DENY_SOURCES_DEFAULT)
    src = str(meta.get("source", "") or "").strip()
    if src in deny_sources:
        return True
    url = str(meta.get("url", "") or "")
    if "n.news.naver.com" in url or "news.naver.com" in url:
        return True
    return False


def _norm01(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


# ====== (너 기존: BOOST/RERANK 등은 여기서 유지해도 OK) ======
BOOST_KEYWORDS = {
    "부산": 0.13,
    "디저트": 0.12,
    "카페": 0.15,
    "맛집": 0.08,
    "두쫀쿠": 0.25,
    "두콘쭈": 0.20,
    "두바이쫀득쿠키": 0.20,
    "쫀득쿠키": 0.15,
    "쿠키": 0.06,
    "팝업": 0.05,
    "센텀": 0.06,
    "광안리": 0.06,
    "서면": 0.05,
    "해운대": 0.05,
    "전포": 0.05,
    "동래": 0.04,
    "남포": 0.04,
}

def _kw_bonus(query: str, meta: Dict[str, Any]) -> float:
    q = (query or "")
    title = str(meta.get("title", "") or "")
    text = str(meta.get("text", "") or "")
    blob = (title + " " + text)
    bonus = 0.0
    for kw, w in BOOST_KEYWORDS.items():
        if kw in q:
            if kw in blob:
                bonus += (w * 1.5)
            else:
                bonus += (w * 0.4)
        else:
            if kw in blob:
                bonus += (w * 0.25)
    return float(bonus)

def _rerank(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    rescored: List[Dict[str, Any]] = []
    for h in hits:
        base = float(h.get("score", 0.0))
        b = _kw_bonus(query, h)
        h["score_rerank"] = base + b
        rescored.append(h)
    rescored.sort(key=lambda x: float(x.get("score_rerank", x.get("score", 0.0))), reverse=True)
    for i, h in enumerate(rescored):
        h["rank"] = i
    return rescored


# ====== filters (너 기존 로직 유지) ======
FILTER_AREA_ALIASES: Dict[str, List[str]] = {
    "광안리": ["광안리", "민락", "금련산"],
    "해운대": ["해운대", "센텀", "센텀시티", "우동", "장산", "재송"],
    "전포": ["전포", "전포동"],
    "서면": ["서면", "부전", "부전역"],
    "남포": ["남포", "남포동", "자갈치", "광복", "국제시장"],
    "동래": ["동래", "온천장", "미남", "사직"],
    "기장": ["기장", "송정", "일광", "정관"],
}

FILTER_TYPE_ALIASES: Dict[str, List[str]] = {
    "베이커리": ["베이커리", "빵", "빵집", "제과", "크로와상", "휘낭시에", "식빵", "바게트", "스콘"],
    "대형카페": ["대형", "넓", "루프탑", "오션뷰", "바다뷰", "뷰", "정원", "테라스", "주차", "층고", "좌석"],
    "디저트": ["디저트", "쿠키", "케이크", "타르트", "마카롱", "젤라또", "푸딩", "아이스크림", "파르페"],
}

def _blob(meta: Dict[str, Any]) -> str:
    title = str(meta.get("title", "") or "")
    text = str(meta.get("text", "") or "")
    url = str(meta.get("url", "") or "")
    shop = str(meta.get("shop", "") or "")  # ✅ NEW: shop도 blob에 포함(필터/매칭에 도움)
    return f"{title} {shop} {text} {url}"

def _canonical_to_tokens(kind: str, canonical: str) -> List[str]:
    if kind == "area":
        return FILTER_AREA_ALIASES.get(canonical, [canonical])
    if kind == "type":
        return FILTER_TYPE_ALIASES.get(canonical, [canonical])
    return [canonical]

def _match_any(blob: str, tokens: List[str]) -> bool:
    return any(t and (t in blob) for t in tokens)

def _passes_filters(meta: Dict[str, Any], filters: Optional[Dict[str, str]]) -> bool:
    if not filters:
        return True
    b = _blob(meta)
    area = str(filters.get("area", "") or "").strip()
    ftype = str(filters.get("type", "") or "").strip()
    if area:
        if not _match_any(b, _canonical_to_tokens("area", area)):
            return False
    if ftype:
        if not _match_any(b, _canonical_to_tokens("type", ftype)):
            return False
    return True

def _relax_filters(filters: Dict[str, str], step: int) -> Dict[str, str]:
    area = str(filters.get("area", "") or "").strip()
    ftype = str(filters.get("type", "") or "").strip()
    if step == 0:
        return {"area": area, "type": ftype} if (area or ftype) else {}
    if step == 1:
        return {"area": area} if area else {}
    if step == 2:
        return {"type": ftype} if ftype else {}
    return {}


# ====== IO helpers ======
def _load_meta_lines(meta_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(meta_path):
        return out
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _resolve_store_paths(store_dir: Optional[str] = None, store_name: Optional[str] = None) -> Dict[str, str]:
    sd = (store_dir or "").strip() or DEFAULT_STORE_DIR
    if store_name:
        sd = os.path.join(sd, store_name)
    return {
        "store_dir": sd,
        "index_path": os.path.join(sd, "index.faiss"),
        "meta_path": os.path.join(sd, "meta.jsonl"),
        "config_path": os.path.join(sd, "config.json"),
    }

def _load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _get_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers import failed")
    key = model_name.strip()
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    m = SentenceTransformer(key)
    _MODEL_CACHE[key] = m
    return m

def _load_index(index_path: str):
    if faiss is None:
        raise RuntimeError("faiss import failed")
    if index_path in _INDEX_CACHE:
        return _INDEX_CACHE[index_path]
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index not found: {index_path}")
    idx = faiss.read_index(index_path)
    _INDEX_CACHE[index_path] = idx
    return idx

def _load_meta(meta_path: str):
    if meta_path in _META_CACHE:
        return _META_CACHE[meta_path]
    lines = _load_meta_lines(meta_path)
    _META_CACHE[meta_path] = lines
    return lines


def retrieve(
    query: str,
    k: int = 4,
    store_dir: Optional[str] = None,
    store_name: Optional[str] = None,
    min_score: float = -1.0,
    filters: Optional[Dict[str, str]] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    paths = _resolve_store_paths(store_dir=store_dir, store_name=store_name)
    index_path = paths["index_path"]
    meta_path = paths["meta_path"]
    config_path = paths["config_path"]

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return []

    cfg = _load_config(config_path)
    model_name = str(cfg.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2")

    try:
        model = _get_model(model_name)
        q_emb = model.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim != 2 or q_emb.shape[0] != 1:
            return []

        index = _load_index(index_path)
        meta_lines = _load_meta(meta_path)

        kk = max(1, int(k))
        search_k = min(max(kk * 10, kk), max(kk, len(meta_lines)))

        D, I = index.search(q_emb, search_k)

        base_filters = dict(filters or {})
        relax_steps = [0, 1, 2, 3] if base_filters else [3]

        final_hits: List[Dict[str, Any]] = []
        final_used_filters: Dict[str, str] = {}

        for step in relax_steps:
            use_filters = _relax_filters(base_filters, step)
            final_used_filters = dict(use_filters)

            hits: List[Dict[str, Any]] = []
            for rank in range(search_k):
                idx = int(I[0][rank])
                score = float(D[0][rank])

                if idx < 0 or idx >= len(meta_lines):
                    continue
                if score < float(min_score):
                    continue

                m = meta_lines[idx] if isinstance(meta_lines[idx], dict) else {}
                if _is_blocked(m):
                    continue
                if not _passes_filters(m, use_filters):
                    continue

                hits.append(
                    {
                        "chunk_id": str(m.get("chunk_id", "")),
                        "doc_id": str(m.get("doc_id", "")),
                        "chunk_idx": int(m.get("chunk_idx", 0) or 0),
                        "url": str(m.get("url", "")),
                        "source": str(m.get("source", "")),
                        "title": str(m.get("title", "")),
                        "fetched_at": str(m.get("fetched_at", "")),
                        "shop": str(m.get("shop", "")),     # ✅ NEW: shop 반환
                        "text": str(m.get("text", "")),
                        "score": _norm01(score),
                        "rank": int(rank),
                    }
                )

                if len(hits) >= max(kk * 5, kk):
                    break

            hits = _rerank(q, hits)
            hits = hits[:kk]

            if hits:
                final_hits = hits
                break

        if final_hits and base_filters:
            for h in final_hits:
                h["filters_used"] = final_used_filters

        return final_hits

    except Exception:
        return []

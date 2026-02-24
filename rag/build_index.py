# rag/build_index.py
from __future__ import annotations

import os
import json
import glob
import re
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


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except Exception:
            return ""


def _extract_text_from_json(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""
    meta = obj.get("meta")
    if isinstance(meta, dict):
        mt = meta.get("text")
        if isinstance(mt, str) and mt.strip():
            return mt.strip()

    t = obj.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    for k in ("content", "body", "article", "본문"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_meta_from_json(obj: Dict[str, Any], path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict):
        m = obj.get("meta")
        if isinstance(m, dict):
            meta.update(m)

        if "title" in obj and "title" not in meta:
            meta["title"] = obj.get("title")
        if "url" in obj and "url" not in meta:
            meta["url"] = obj.get("url")

    meta.setdefault("id", "")
    meta.setdefault("url", "")
    meta.setdefault("title", os.path.basename(path))
    meta.setdefault("source", "")
    meta.setdefault("fetched_at", "")
    meta.setdefault("shop", "")  # ✅ NEW (meta 기본 키)
    meta["path"] = path
    return meta


def _simple_chunk(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        ch = text[i:j].strip()
        if ch:
            out.append(ch)
        if j >= n:
            break
        i = max(0, j - overlap)
    return out


# ------------------------------------------------------------
# ✅ shop extraction heuristic (index-time)
# ------------------------------------------------------------
_STOP_SHOP = {
    "부산", "카페", "맛집", "추천", "후기", "리뷰", "방문기", "솔직후기",
    "디저트", "베이커리", "빵집", "제과", "브런치", "핫플", "웨이팅"
}

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def _shop_from_title(title: str) -> str:
    t = _clean_text(title)
    if not t:
        return ""

    if "|" in t:
        left = t.split("|", 1)[0].strip()
        left = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", left).strip()
        if 2 <= len(left) <= 30 and left not in _STOP_SHOP:
            return left

    if ":" in t:
        after = t.split(":")[-1].strip()
        after = re.sub(r"(솔직후기|후기|리뷰|방문기)$", "", after).strip()
        if 2 <= len(after) <= 30 and after not in _STOP_SHOP:
            return after

    m = re.search(r"([가-힣A-Za-z0-9_]{2,25})\s*(?:솔직후기|후기|리뷰|방문기)\b", t)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    return ""

def _shop_from_text(text: str) -> str:
    s = _clean_text(text)
    if not s:
        return ""

    m = re.search(r"(?:카페|베이커리|제과점)\s*([가-힣A-Za-z0-9_]{2,25})", s)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    m = re.search(r"([가-힣A-Za-z0-9_]{2,25})\s*\((?:카페|베이커리|제과점)\)", s)
    if m:
        cand = m.group(1).strip()
        if cand and cand not in _STOP_SHOP:
            return cand

    return ""

def _extract_shop(meta: Dict[str, Any], full_text: str) -> str:
    # 1) crawl 단계에서 이미 들어온 값이 있으면 우선
    shop = str(meta.get("shop", "") or "").strip()
    if shop and shop not in _STOP_SHOP:
        return shop

    # 2) title 기반
    shop = _shop_from_title(str(meta.get("title", "") or ""))
    if shop:
        return shop

    # 3) 본문 기반 (너무 길면 앞부분만)
    head = (full_text or "")[:2500]
    shop = _shop_from_text(head)
    if shop:
        return shop

    return ""


def build_index(
    data_dir: str,
    store_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    store_name: Optional[str] = None,
    chunk_size: int = 900,
    overlap: int = 120,
) -> Dict[str, Any]:
    """
    ✅ data_dir 아래의 *.json / *.md / *.txt 를 모아서
       chunk -> embedding -> faiss index(IP) + meta.jsonl 생성

    ✅ NEW: meta.jsonl에 shop 필드 저장
    """
    if faiss is None:
        return {"ok": False, "error": "faiss not installed/import failed"}
    if SentenceTransformer is None:
        return {"ok": False, "error": "sentence-transformers import failed"}

    data_dir = str(data_dir)
    store_dir = str(store_dir)
    if store_name:
        store_dir = os.path.join(store_dir, store_name)

    os.makedirs(store_dir, exist_ok=True)

    # 1) 파일 수집
    paths: List[str] = []
    paths += glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)

    docs: List[Dict[str, Any]] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()

        if ext in (".md", ".txt"):
            text = _read_text(p).strip()
            if not text:
                continue
            docs.append(
                {
                    "id": os.path.splitext(os.path.basename(p))[0],
                    "meta": {"url": "", "title": os.path.basename(p), "source": "manual", "fetched_at": "", "path": p, "shop": ""},
                    "text": text,
                }
            )
            continue

        if ext == ".json":
            raw = _read_text(p)
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            text = _extract_text_from_json(obj)
            if not text.strip():
                continue

            meta = _extract_meta_from_json(obj, p)
            doc_id = str(meta.get("id") or os.path.splitext(os.path.basename(p))[0])

            # ✅ NEW: 여기서 shop 확정
            meta["shop"] = _extract_shop(meta, text)

            docs.append({"id": doc_id, "meta": meta, "text": text})

    if not docs:
        return {
            "ok": False,
            "error": "no_docs_loaded_from_data_dir",
            "data_dir": data_dir,
            "hint": "rag/data에 json/md/txt가 있어도 text가 비어있으면 스킵됨",
            "num_docs": 0,
            "num_chunks": 0,
        }

    # 2) 청킹 + meta 라인
    meta_lines: List[Dict[str, Any]] = []
    all_chunks: List[str] = []

    for d in docs:
        doc_id = str(d["id"])
        meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
        text = str(d.get("text", ""))

        chunks = _simple_chunk(text, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(chunks):
            chunk_id = f"{doc_id}::c{ci}"
            meta_lines.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_idx": ci,
                    "url": str(meta.get("url", "")),
                    "source": str(meta.get("source", "")),
                    "title": str(meta.get("title", "")),
                    "fetched_at": str(meta.get("fetched_at", "")),
                    "shop": str(meta.get("shop", "")),   # ✅ NEW
                    "text": ch,
                }
            )
            all_chunks.append(ch)

    if not all_chunks:
        return {
            "ok": False,
            "error": "no_chunks_after_chunking",
            "data_dir": data_dir,
            "num_docs": len(docs),
            "num_chunks": 0,
        }

    # 3) 임베딩
    model = SentenceTransformer(model_name)
    emb = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[0] != len(all_chunks):
        return {
            "ok": False,
            "error": "bad_embeddings_shape",
            "shape": list(getattr(emb, "shape", [])),
            "num_chunks": len(all_chunks),
        }

    dim = int(emb.shape[1])

    # 4) FAISS index (IP = cosine with normalized embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.jsonl")
    config_path = os.path.join(store_dir, "config.json")

    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta_lines:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": model_name, "dim": dim, "num_docs": len(docs), "num_chunks": len(all_chunks)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "ok": True,
        "data_dir": data_dir,
        "store_dir": store_dir,
        "index_path": index_path,
        "meta_path": meta_path,
        "config_path": config_path,
        "num_docs": len(docs),
        "num_chunks": len(all_chunks),
        "dim": dim,
        "model_name": model_name,
    }

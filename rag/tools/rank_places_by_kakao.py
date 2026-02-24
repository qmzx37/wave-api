# rag/tools/rank_places_by_kakao.py
from __future__ import annotations
import os, json
from typing import Any, Dict, List

IN_PATH  = os.getenv("PLACES_IN",  r"rag\data\places\places_enriched.jsonl")
OUT_PATH = os.getenv("PLACES_OUT", r"rag\data\places\places_ranked.jsonl")
MIN_REV  = int(os.getenv("MIN_REV", "3"))   # 후기수 최소 컷 (0~)
LIMIT    = int(os.getenv("LIMIT", "0"))     # 0이면 전체

def _read(path: str) -> List[Dict[str, Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8-sig") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows

def _write(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def score(r: Dict[str, Any]) -> float:
    # 후기수 우선, 별점은 동률 깨기
    rc = r.get("kakao_review_count")
    rt = r.get("kakao_rating")
    try: rc = int(rc) if rc is not None else 0
    except: rc = 0
    try: rt = float(rt) if rt is not None else 0.0
    except: rt = 0.0
    return float(rc) + 0.05 * float(rt)

def main():
    rows = _read(IN_PATH)

    # 후기수 없는 애들은 뒤로 보내되, 완전 제거는 일단 안 함(나중에 파싱 개선 가능)
    for r in rows:
        if r.get("kakao_review_count") is None:
            r["kakao_review_count"] = 0
        if r.get("kakao_rating") is None:
            r["kakao_rating"] = 0.0

    # 최소 후기수 컷
    filtered = []
    for r in rows:
        try:
            if int(r.get("kakao_review_count") or 0) >= MIN_REV:
                filtered.append(r)
        except:
            pass

    filtered.sort(key=score, reverse=True)

    if LIMIT > 0:
        filtered = filtered[:LIMIT]

    _write(OUT_PATH, filtered)

    print(f"[DONE] wrote: {os.path.abspath(OUT_PATH)}")
    print(f"[STAT] in={len(rows)} kept={len(filtered)} min_rev={MIN_REV} limit={LIMIT}")

    # top 10 프린트
    print("\n[TOP10]")
    for i, r in enumerate(filtered[:10], start=1):
        print(f"{i:02d}. {r.get('name','')}  rev={r.get('kakao_review_count')}  rating={r.get('kakao_rating')}  url={r.get('url','')}")

if __name__ == "__main__":
    main()

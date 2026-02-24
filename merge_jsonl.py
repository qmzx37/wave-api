# merge_jsonl.py
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Any, Iterable, Tuple

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

def pick_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    # 1순위: doc_id
    doc_id = str(rec.get("doc_id", "") or "").strip()
    if doc_id:
        return ("doc_id", doc_id)

    # 2순위: url
    url = str(rec.get("url", "") or "").strip()
    if url:
        return ("url", url)

    # 3순위: (shop + address)
    shop = str(rec.get("shop", "") or rec.get("place_name", "") or "").strip()
    addr = str(rec.get("address", "") or rec.get("address_name", "") or rec.get("road_address_name", "") or "").strip()
    if shop or addr:
        return ("shop_addr", f"{shop}|{addr}")

    # 최후: 전체 덤프(비추천이지만 안전장치)
    return ("dump", json.dumps(rec, ensure_ascii=False, sort_keys=True)[:2000])

def main():
    # ✅ 너가 업로드한 두 파일 경로(그대로 사용 가능)
    in1 = r"/mnt/data/kakao_places_2026-02-01.jsonl"
    in2 = r"/mnt/data/kakao_places_gijang_2026-02-01.jsonl"
    out = r"/mnt/data/kakao_places_merged_2026-02-01.jsonl"

    # ✅ 네 PC에서 돌릴 땐 이 3줄만 너 경로로 바꾸면 됨:
    # in1 = r"rag\data\kakao_places\kakao_places_2026-02-01.jsonl"
    # in2 = r"rag\data\kakao_places\kakao_places_gijang_2026-02-01.jsonl"
    # out = r"rag\data\kakao_places\kakao_places_2026-02-01_merged.jsonl"

    seen = set()
    kept = 0
    dup = 0
    src_counter = Counter()

    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, "w", encoding="utf-8") as w:
        for p in [in1, in2]:
            for rec in iter_jsonl(p):
                # source 통계(있으면)
                src = str(rec.get("source", "") or rec.get("meta", {}).get("source", "") or "")
                if src:
                    src_counter[src] += 1

                _, key = pick_key(rec)
                if not key:
                    continue
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)

                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[OK] merged -> {out}")
    print(f"[STAT] kept={kept} dup_dropped={dup} unique_keys={len(seen)}")
    if src_counter:
        print("[STAT] sources:")
        for k, v in src_counter.most_common(10):
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()

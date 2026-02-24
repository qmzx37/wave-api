from __future__ import annotations

import os
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt

# -------------------------
# 설정
# -------------------------
DEFAULT_LOG = r"C:\llm\train\emotion_log.jsonl"
NUM_ACTIONS = 12

# action_id 찾을 때 후보 경로들(버전 흔들림 방어)
# 1) meta.wave.action.id
# 2) meta.action.id
# 3) wave.action.id
# 4) action_id (최상위)
ACTION_ID_PATHS = [
    ("meta", "wave", "action", "id"),
    ("meta", "action", "id"),
    ("wave", "action", "id"),
    ("action_id",),
]

# goal/style 찾을 때 후보 경로들
GOAL_PATHS = [
    ("meta", "wave", "action", "goal"),
    ("meta", "action", "goal"),
    ("wave", "action", "goal"),
    ("goal",),
]
STYLE_PATHS = [
    ("meta", "wave", "action", "style"),
    ("meta", "action", "style"),
    ("wave", "action", "style"),
    ("style",),
]


# -------------------------
# 유틸
# -------------------------
def _safe_get(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _first_found(d: Dict[str, Any], paths: List[Tuple[str, ...]]) -> Any:
    for p in paths:
        v = _safe_get(d, p)
        if v is not None:
            return v
    return None

def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        # "3" 같은 문자열도 처리
        i = int(float(x))
        return i
    except Exception:
        return None

def _to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log not found: {path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # 깨진 줄은 스킵
                continue
    return out


# -------------------------
# 메인 분석
# -------------------------
def main():
    log_path = os.environ.get("LOG_PATH", DEFAULT_LOG)
    print("LOG_PATH:", log_path)

    recs = read_jsonl(log_path)
    print("records:", len(recs))

    # 전체 action histogram
    c_all = Counter()

    # (goal, style)별 action histogram
    c_by_pair: dict[Tuple[str, str], Counter] = defaultdict(Counter)

    # 디버그: action_id 없는 레코드 수
    missing_action = 0

    for r in recs:
        aid_raw = _first_found(r, ACTION_ID_PATHS)
        aid = _to_int(aid_raw)

        if aid is None:
            missing_action += 1
            continue

        if aid < 0 or aid >= NUM_ACTIONS:
            # 범위를 벗어나면 일단 스킵 (원하면 여기서 clamp해도 됨)
            continue

        c_all[aid] += 1

        goal = _to_str(_first_found(r, GOAL_PATHS)) or "UNKNOWN"
        style = _to_str(_first_found(r, STYLE_PATHS)) or "UNKNOWN"
        c_by_pair[(goal, style)][aid] += 1

    print("missing_action:", missing_action)
    total = sum(c_all.values())
    print("total_actions_counted:", total)

    if total == 0:
        print("❌ action_id가 로그에서 안 잡혔어.")
        print("→ 지금 api.py에서 append_log에 action을 저장하는지부터 확인해야 함.")
        return

    # -------------------------
    # 1) 텍스트 요약 출력
    # -------------------------
    print("\n=== Action distribution (counts / pct) ===")
    for a in range(NUM_ACTIONS):
        cnt = c_all.get(a, 0)
        pct = (cnt / total * 100.0) if total else 0.0
        print(f"action {a:02d}: {cnt:6d}  ({pct:5.1f}%)")

    # 상위 pair 몇 개만 출력(데이터 많으면 보기 힘드니)
    print("\n=== Top (goal, style) pairs by volume ===")
    pair_totals = sorted(((k, sum(v.values())) for k, v in c_by_pair.items()), key=lambda x: x[1], reverse=True)
    for (goal, style), cnt in pair_totals[:10]:
        print(f"{goal:10s} / {style:10s} : {cnt}")

    # -------------------------
    # 2) 전체 히스토그램
    # -------------------------
    xs = list(range(NUM_ACTIONS))
    ys = [c_all.get(i, 0) for i in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.title("Action ID Histogram (All)")
    plt.xlabel("action_id (0..11)")
    plt.ylabel("count")
    plt.xticks(xs)
    plt.tight_layout()

    # -------------------------
    # 3) (goal, style)별 히스토그램(상위 4개만)
    # -------------------------
    top_pairs = [k for k, _ in pair_totals[:4]]
    for (goal, style) in top_pairs:
        c = c_by_pair[(goal, style)]
        ys2 = [c.get(i, 0) for i in xs]
        plt.figure()
        plt.bar(xs, ys2)
        plt.title(f"Action Histogram: {goal} / {style}")
        plt.xlabel("action_id (0..11)")
        plt.ylabel("count")
        plt.xticks(xs)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

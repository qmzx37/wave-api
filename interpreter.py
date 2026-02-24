# interpreter.py
from __future__ import annotations

from typing import Dict, Tuple

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]
STATE_KEYS = ["F", "A", "D", "J"]


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def top_state(axes: Dict[str, float]) -> Tuple[str, float]:
    best_k = "J"
    best_v = -1.0
    for k in STATE_KEYS:
        v = float(axes.get(k, 0.0))
        if v > best_v:
            best_k, best_v = k, v
    return best_k, clamp01(best_v)


def classify_mode(axes: Dict[str, float]) -> str:
    """
    4클러스터 모드(JOY/ANGER/SADNESS/ANXIETY) + NEUTRAL/MIXED 보조
    - ANXIETY는 F와 T가 함께 높을 때 우선
    """
    axes = {k: clamp01(axes.get(k, 0.0)) for k in KEYS}
    F, A, D, J, T = axes["F"], axes["A"], axes["D"], axes["J"], axes["T"]

    # 불안(ANXIETY) 우선 규칙: 공포+긴장 조합
    if (F >= 0.45 and T >= 0.45) or (T >= 0.70 and F >= 0.25):
        return "ANXIETY"

    k, v = top_state(axes)

    if v < 0.25 and T < 0.35:
        return "NEUTRAL"

    # 혼합: 상위 2개가 비슷하면 MIXED
    vals = sorted([("F", F), ("A", A), ("D", D), ("J", J)], key=lambda x: x[1], reverse=True)
    if vals[0][1] - vals[1][1] <= 0.08 and vals[0][1] >= 0.35:
        return "MIXED"

    if k == "J":
        return "JOY"
    if k == "A":
        return "ANGER"
    if k == "D":
        return "SADNESS"
    if k == "F":
        return "ANXIETY" if T >= 0.35 else "ANXIETY"  # 공포는 일단 불안으로 묶자

    return "NEUTRAL"


def derive_tag(axes: Dict[str, float]) -> str:
    """
    다음 채팅 추천을 위한 태그:
    - comfort : 위로/정리/안정 필요
    - vent    : 감정 배출(분노/억울)
    - explore : 호기심 확장
    - decide  : 욕심/선택/집착(뭘 하고 싶다)
    - neutral : 중립
    """
    axes = {k: clamp01(axes.get(k, 0.0)) for k in KEYS}
    A, D, F, J = axes["A"], axes["D"], axes["F"], axes["J"]
    C, G, T, R = axes["C"], axes["G"], axes["T"], axes["R"]

    if (D >= 0.45) or (F >= 0.45 and R <= 0.25) or (T >= 0.65):
        return "comfort"
    if A >= 0.55:
        return "vent"
    if C >= 0.45 and G < 0.35:
        return "explore"
    if G >= 0.45:
        return "decide"
    if J >= 0.55 and R >= 0.45:
        return "uplift"
    return "neutral"

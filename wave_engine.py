# wave_engine.py
from __future__ import annotations
from typing import Dict
import time
import math

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))

def init_wave_state() -> Dict[str, float]:
    # psi_* 는 "누적된 파동의 상태(관성)"
    state = {f"psi_{k}": 0.0 for k in KEYS}
    state["session_t0"] = time.time()
    state["last_t"] = time.time()
    state["freq"] = 1.0       # 파동 속도(상대값)
    state["damping"] = 0.2    # 감쇠(상대값)
    return state

def update_wave_state(state: Dict[str, float], axes: Dict[str, float], now: float | None = None) -> Dict[str, float]:
    now = now or time.time()
    dt = max(0.0, now - float(state.get("last_t", now)))
    state["last_t"] = now

    # 관성(EMA) 계수: T가 높으면 민감(빠르게 반응), R이 높으면 둔감(느리게 반응)
    T = clamp01(axes.get("T", 0.0))
    R = clamp01(axes.get("R", 0.0))
    alpha = clamp01(0.25 + 0.35*T - 0.20*R)  # 0~1

    for k in KEYS:
        prev = float(state.get(f"psi_{k}", 0.0))
        cur = clamp01(axes.get(k, 0.0))
        state[f"psi_{k}"] = clamp01((1 - alpha) * prev + alpha * cur)

    # 파동 파라미터(네 해석 반영)
    # - T↑ -> freq↑ (빠른 주파수)
    # - R↑ -> damping↑ (가라앉는 힘), freq↓(느려짐)
    state["freq"] = 0.8 + 1.6*T - 0.6*R
    state["damping"] = 0.1 + 0.9*R

    # 30~40분 후 “종료 유도” 느낌: R을 서서히 올리고 T를 살짝 내림
    elapsed = now - float(state.get("session_t0", now))
    if elapsed >= 30 * 60:  # 30분
        bump = clamp01((elapsed - 30*60) / (10*60)) * 0.08  # 30~40분 사이 최대 +0.08
        state["psi_R"] = clamp01(state["psi_R"] + bump)
        state["psi_T"] = clamp01(state["psi_T"] - bump * 0.8)

    return state

def summarize_wave(state: Dict[str, float]) -> Dict[str, float | str]:
    psiT = clamp01(state.get("psi_T", 0.0))
    psiR = clamp01(state.get("psi_R", 0.0))

    if psiT >= 0.65:
        pace = "fast"
    elif psiR >= 0.65:
        pace = "slow"
    else:
        pace = "mid"

    # 지배 상태(4개)만 뽑아줌
    dom = max(["F", "A", "D", "J"], key=lambda k: float(state.get(f"psi_{k}", 0.0)))
    dom_val = float(state.get(f"psi_{dom}", 0.0))

    return {
        "pace": pace,
        "dominant": dom,
        "dominant_val": dom_val,
        "freq": float(state.get("freq", 1.0)),
        "damping": float(state.get("damping", 0.2)),
        "psi_T": psiT,
        "psi_R": psiR,
    }

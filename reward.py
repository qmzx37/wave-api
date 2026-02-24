# reward.py  (AMPLIFY 완화 + action entropy diversity bonus + anti-collapse(goal repeat) + amplify_bias 반영)
from __future__ import annotations

from typing import Dict, Any, Tuple, List
import math

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]
Axis = Dict[str, float]
ActionPair = Tuple[str, str]  # (goal, style)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def clamp01(x: Any) -> float:
    x = _f(x, 0.0)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def norm_axes(a: Dict[str, Any]) -> Axis:
    return {k: clamp01(a.get(k, 0.0)) for k in KEYS}


def d(curr: Axis, prev: Axis, k: str) -> float:
    return float(curr.get(k, 0.0) - prev.get(k, 0.0))


def get_phase(wave: Dict[str, Any]) -> str:
    wave = wave or {}
    sess = wave.get("session", {})
    if isinstance(sess, dict):
        p = sess.get("phase")
        if p:
            return str(p)

        if "t" in sess and any(k in sess for k in ["t_max", "tMax", "max", "T_MAX"]):
            try:
                t = int(sess.get("t", 0))
                t_max = sess.get("t_max", None)
                if t_max is None:
                    t_max = sess.get("tMax", None)
                if t_max is None:
                    t_max = sess.get("max", None)
                if t_max is None:
                    t_max = sess.get("T_MAX", None)
                t_max = int(t_max) if t_max is not None else 1

                ratio = float(t) / float(max(1, t_max))
                if ratio < 0.40:
                    return "ENGAGE"
                if ratio < 0.80:
                    return "SETTLE"
                return "CLOSE"
            except Exception:
                pass

    pace = str(wave.get("pace", "mid"))
    drift = float(wave.get("drift", 0.0))
    if pace == "high" or drift >= 0.55:
        return "ENGAGE"
    if pace == "low" and drift < 0.20:
        return "CLOSE"
    return "SETTLE"


def is_direct(style: str) -> bool:
    return str(style).upper() in ("DIRECT", "D")


def is_short(style: str) -> bool:
    return str(style).upper() in ("SHORT", "S")


def is_normal(style: str) -> bool:
    return str(style).upper() in ("NORMAL", "N")


# ------------------------------------------------------------
# diversity bonus helpers (학습용)
# ------------------------------------------------------------
def _entropy_of_action_ids(ids: List[int], num_actions: int = 12) -> float:
    """0..log(num_actions)"""
    if not ids:
        return 0.0
    counts = [0] * int(num_actions)
    for a in ids:
        try:
            ai = int(a)
        except Exception:
            continue
        if 0 <= ai < num_actions:
            counts[ai] += 1
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        ent -= p * math.log(max(1e-12, p))
    return float(ent)


def _entropy_01(ids: List[int], num_actions: int = 12) -> float:
    """0..1 정규화"""
    ent = _entropy_of_action_ids(ids, num_actions=num_actions)
    max_ent = math.log(float(num_actions))
    if max_ent <= 0:
        return 0.0
    return float(max(0.0, min(1.0, ent / max_ent)))


# ------------------------------------------------------------
# (NEW) goal repeat penalty helpers
# ------------------------------------------------------------
def _repeat_ratio_goals(recent_action_ids: List[int], num_actions: int = 12) -> float:
    """
    recent_action_ids -> action_pair(goal, style)로 goal 시퀀스 만들고
    '가장 많이 나온 goal' 비율 반환 (0..1)
    """
    if not recent_action_ids:
        return 0.0
    # action_space import는 reward.py에서 직접 하면 순환될 수 있으니,
    # offline_env가 session에 recent_goal_ids 같은 걸 넣어주면 더 좋지만
    # 지금은 action_id만으로 "반복"을 보려면 action_id 반복률로 대체.
    # => 여기서는 action_id 반복률로 구현(간단/안전)
    ids = [int(x) for x in recent_action_ids if isinstance(x, (int, float, str))]
    if not ids:
        return 0.0
    most = max(set(ids), key=ids.count)
    return float(ids.count(most) / float(len(ids)))


# ------------------------------------------------------------
# 핵심 reward
# ------------------------------------------------------------
def compute_reward(
    prev_axes_in: Dict[str, Any],
    prev_wave_in: Dict[str, Any],
    action: ActionPair,
    curr_axes_in: Dict[str, Any],
    curr_wave_in: Dict[str, Any],
) -> float:
    """
    ✅ 목표:
      - 기본: "초반 ENGAGE는 C/G, 이후 SETTLE/CLOSE는 R 수렴"
      - ✅ AMPLIFY 과몰림 방지: AMPLIFY 달달함을 줄이고, J가 높을 때만 약하게 밀기
      - ✅ 다양성: entropy bonus는 살짝 강화(하지만 너무 크면 망함)
      - ✅ 반복 페널티: recent_action_ids가 한 액션에 몰리면 감점
    """
    prev_axes = norm_axes(prev_axes_in)
    curr_axes = norm_axes(curr_axes_in)

    prev_wave = prev_wave_in if isinstance(prev_wave_in, dict) else {}
    curr_wave = curr_wave_in if isinstance(curr_wave_in, dict) else {}

    goal, style = action
    goal = str(goal).upper()
    style = str(style).upper()

    phase = get_phase(curr_wave)

    # 변화량
    dF = d(curr_axes, prev_axes, "F")
    dA = d(curr_axes, prev_axes, "A")
    dD = d(curr_axes, prev_axes, "D")
    dJ = d(curr_axes, prev_axes, "J")
    dC = d(curr_axes, prev_axes, "C")
    dG = d(curr_axes, prev_axes, "G")
    dT = d(curr_axes, prev_axes, "T")
    dR = d(curr_axes, prev_axes, "R")

    # 현재값
    A = curr_axes["A"]
    Dv = curr_axes["D"]
    Jv = curr_axes["J"]
    Tv = curr_axes["T"]
    Rv = curr_axes["R"]

    # --------------------------------------------------------
    # 1) phase 기반 shaping
    # --------------------------------------------------------
    if phase == "ENGAGE":
        w_CG = 1.30
        w_R = 0.30
        w_T_pen = 0.40
        w_AD_pen = 0.35
    elif phase == "SETTLE":
        w_CG = 0.45
        w_R = 1.10
        w_T_pen = 0.75
        w_AD_pen = 0.50
    else:  # CLOSE
        w_CG = 0.20
        w_R = 1.40
        w_T_pen = 0.95
        w_AD_pen = 0.60

    reward = 0.0
    reward += w_CG * (0.7 * dC + 0.7 * dG)
    reward += w_R * (0.9 * dR)
    reward += -w_T_pen * (0.9 * dT)
    reward += -w_AD_pen * (0.5 * dA + 0.5 * dD)

    # --------------------------------------------------------
    # 2) goal별 보상
    # --------------------------------------------------------
    if goal == "STABILIZE":
        reward += 1.00 * dR - 0.90 * dT - 0.25 * dA - 0.25 * dF

    elif goal == "CLARIFY":
        reward += -0.85 * dA - 0.55 * dF - 0.35 * dT + 0.15 * dR

    elif goal == "MOTION":
        reward += -1.10 * dD + 0.35 * dR + 0.15 * dG + 0.10 * dC

    elif goal == "AMPLIFY":
        # ✅ [FIX] AMPLIFY 달달함 줄이기
        # - 기존(너 코드): 1.35*dJ + 0.45*dC + 0.06
        # - 수정: 1.05*dJ + 0.30*dC + (고정보너스 제거)
        reward += 1.05 * dJ + 0.30 * dC - 0.45 * max(0.0, dT)

        # ✅ 고정 보너스 제거 (이게 AMP 몰림 촉발 1순위)
        # reward += 0.06  # 제거

    # --------------------------------------------------------
    # 2.5) ✅ J가 높은 상태면 AMPLIFY를 "조금만" 밀어주기
    # - 기존 0.18 -> 0.08로 낮춤
    # --------------------------------------------------------
    if Jv >= 0.60:
        if goal == "AMPLIFY":
            reward += 0.08
        if goal == "MOTION":
            reward -= 0.04

    # --------------------------------------------------------
    # 2.6) amplify_bias 반영(상한 낮추기)
    # - 0.25 상한 -> 0.12 상한
    # --------------------------------------------------------
    try:
        sess = curr_wave.get("session", {})
        if isinstance(sess, dict):
            bias = float(sess.get("amplify_bias", 0.0))
            bias = float(max(0.0, min(0.12, bias)))
            if goal == "AMPLIFY":
                reward += bias
    except Exception:
        pass

    # --------------------------------------------------------
    # 3) style 보정
    # --------------------------------------------------------
    if is_direct(style):
        if A >= 0.60 and Rv <= 0.25:
            reward += 0.25
            if dT > 0.02:
                reward -= 0.20
        else:
            reward -= 0.20

    elif is_short(style):
        if Tv >= 0.55:
            reward += 0.10
        elif Rv >= 0.65 and phase != "ENGAGE":
            reward -= 0.08

    elif is_normal(style):
        reward += 0.02

    # --------------------------------------------------------
    # 3.5) ✅ diversity bonus (엔트로피) — "살짝" 강화
    # - alpha 기본 0.04 -> 0.06 권장 (너무 세면 랜덤됨)
    # --------------------------------------------------------
    try:
        sess = curr_wave.get("session", {})
        if isinstance(sess, dict) and bool(sess.get("enable_diversity_bonus", False)):
            ids = sess.get("recent_action_ids", [])
            if isinstance(ids, list) and ids:
                k = int(sess.get("diversity_k", 12))
                ids_tail = ids[-k:] if k > 0 else ids
                ent01 = _entropy_01([int(x) for x in ids_tail], num_actions=12)

                alpha = float(sess.get("diversity_alpha", 0.06))  # ✅ default up
                alpha = float(max(0.0, min(0.12, alpha)))         # ✅ 상한

                reward += alpha * ent01
    except Exception:
        pass

    # --------------------------------------------------------
    # 3.6) ✅ 반복 페널티: recent_action_ids가 한 액션에 너무 몰리면 감점
    # - AMP 몰림 같은 "정책 붕괴"를 직접 꺾는 장치
    # --------------------------------------------------------
    try:
        sess = curr_wave.get("session", {})
        if isinstance(sess, dict):
            ids = sess.get("recent_action_ids", [])
            if isinstance(ids, list) and len(ids) >= 6:
                rep = _repeat_ratio_goals(ids, num_actions=12)  # 사실상 action repeat ratio
                # rep가 0.80 이상이면 "거의 한 행동만" -> 페널티
                if rep >= 0.80:
                    # ✅ 너무 세면 학습 불안정 -> 0.08~0.20 사이 추천
                    reward -= 0.14 * (rep - 0.80) / 0.20  # rep=1.0이면 -0.14
    except Exception:
        pass

    # --------------------------------------------------------
    # 4) 안전장치
    # --------------------------------------------------------
    if Tv >= 0.85:
        reward -= 0.60
    if A >= 0.85:
        reward -= 0.45
    if Dv >= 0.85:
        reward -= 0.45

    # clip
    if reward > 3.0:
        reward = 3.0
    if reward < -3.0:
        reward = -3.0

    return float(reward)

# offline_env.py  (통째본: 12-action + obs14 + reward.compute_reward 연동 + diversity bonus + amplify_bias)
from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Deque, Tuple
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ✅ 12-action space 연결
from action_space import NUM_ACTIONS, action_pair

# ✅ reward 분리(핵심)
from reward import compute_reward

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]


# ============================================================
# obs dtype/범위/shape 강제 유틸
# ============================================================
def _to_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _clip_box(obs: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(obs, low, high).astype(np.float32, copy=False)


def clamp01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


# ============================================================
# JSONL 로드
# ============================================================
def read_jsonl(path: str) -> List[dict]:
    if not isinstance(path, (str, bytes, os.PathLike)):
        raise TypeError(f"log_path must be a path string, got: {type(path)} -> {path!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LOG_PATH not found: {path}")

    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    return records


def axes_from_record(r: dict) -> Dict[str, float]:
    labels = r.get("labels") or r.get("axes") or {}
    out: Dict[str, float] = {}
    for k in KEYS:
        out[k] = clamp01(labels.get(k, 0.0))
    return out


def axes8_to_dict(v8: np.ndarray) -> Dict[str, float]:
    v8 = np.asarray(v8, dtype=np.float32).reshape((8,))
    return {k: float(v8[i]) for i, k in enumerate(KEYS)}


# ============================================================
# drift / loop 최소 정의(환경 내부 계산)
# ============================================================
def dominant_axis_from_state(state8: np.ndarray) -> str:
    idx = int(np.argmax(state8))
    return KEYS[idx]


def calc_drift(prev8: np.ndarray, cur8: np.ndarray) -> float:
    return float(np.mean(np.abs(cur8 - prev8)))


def calc_loop_score(dom_hist: Deque[str], drift_hist: Deque[float]) -> float:
    if len(dom_hist) < 6:
        return 0.0

    k = 6
    dom_recent = list(dom_hist)[-k:]
    drift_recent = list(drift_hist)[-k:]

    most = max(set(dom_recent), key=dom_recent.count)
    ratio = dom_recent.count(most) / float(k)

    drift_mean = float(np.mean(drift_recent))
    low_drift_factor = max(0.0, 0.25 - drift_mean) / 0.25  # 0..1

    score = ratio * (0.5 + 0.5 * low_drift_factor)
    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# wave(6) feature 유틸  (obs14를 0~1로 맞추기)
# ============================================================
def _signed_to_01(x: float) -> float:
    """-1..1 -> 0..1 스케일"""
    return clamp01((float(x) + 1.0) * 0.5)


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


def pace_from_TR(T: float, R: float) -> str:
    if T >= 0.65:
        return "high"
    if R >= 0.65:
        return "low"
    return "mid"


def loop_kind_from(dom_axis: str, loop_score: float, drift: float) -> str:
    # 아주 단순한 라벨링(Reward 힌트용)
    if loop_score >= 0.75 and drift <= 0.18:
        if dom_axis == "A":
            return "anger_loop"
        if dom_axis == "D":
            return "sad_loop"
    return "none"


# ============================================================
# NoiEOfflineEnv
# ============================================================
class NoiEOfflineEnv(gym.Env):
    """
    ✅ 관측(obs) = 14 dim
      [0..7]   8축 (F,A,D,J,C,G,T,R)
      [8]      drift            (0..1)
      [9]      loop_score       (0..1)
      [10]     dom_value        (0..1)
      [11]     dom_delta_01     (0..1)
      [12]     trend_sign_01    (0..1)
      [13]     rt_gap_01        (0..1)

    ✅ 행동(action) = 12
    ✅ 보상(reward) = reward.compute_reward(...) 사용

    ✅ diversity bonus wiring
    ✅ amplify_bias wiring (session["amplify_bias"])
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Any = None):
        super().__init__()

        if env_config is None:
            env_config = {}

        # RLlib EnvContext 대응
        try:
            log_path = env_config.get("log_path")
            window = int(env_config.get("window", 3000))
            episode_len = int(env_config.get("episode_len", 20))
            loop_window = int(env_config.get("loop_window", 12))
            reward_clip = float(env_config.get("reward_clip", 1.0))

            # ✅ diversity 옵션
            enable_diversity_bonus = bool(env_config.get("enable_diversity_bonus", True))
            diversity_k = int(env_config.get("diversity_k", 12))
            diversity_alpha = float(env_config.get("diversity_alpha", 0.06))  # ✅ default slightly higher

            # ✅ [FIX] AMPLIFY 선호도 기본값 낮추기 (AMP 몰림 방지)
            # 기존 0.08 -> 0.03 권장 (0.00~0.05 사이)
            amplify_bias = float(env_config.get("amplify_bias", 0.03))

            # ✅ (선택) episode마다 bias 흔들어서 “AMP만 정답” 방지
            jitter_amplify_bias = bool(env_config.get("jitter_amplify_bias", True))
            jitter_range = float(env_config.get("amplify_bias_jitter", 0.02))  # ±0.02

        except Exception:
            env_config = dict(env_config)
            log_path = env_config.get("log_path")
            window = int(env_config.get("window", 3000))
            episode_len = int(env_config.get("episode_len", 20))
            loop_window = int(env_config.get("loop_window", 12))
            reward_clip = float(env_config.get("reward_clip", 1.0))

            enable_diversity_bonus = bool(env_config.get("enable_diversity_bonus", True))
            diversity_k = int(env_config.get("diversity_k", 12))
            diversity_alpha = float(env_config.get("diversity_alpha", 0.06))

            amplify_bias = float(env_config.get("amplify_bias", 0.03))
            jitter_amplify_bias = bool(env_config.get("jitter_amplify_bias", True))
            jitter_range = float(env_config.get("amplify_bias_jitter", 0.02))

        if not log_path:
            log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")

        self.log_path = str(log_path)
        self.window = max(10, int(window))
        self.episode_len = max(5, int(episode_len))
        self.loop_window = max(6, int(loop_window))
        self.reward_clip = float(max(0.1, reward_clip))

        # ✅ diversity 설정 저장
        self.enable_diversity_bonus = bool(enable_diversity_bonus)
        self.diversity_k = max(2, int(diversity_k))
        self.diversity_alpha = float(np.clip(diversity_alpha, 0.0, 0.12))  # ✅ cap

        # ✅ amplify bias 저장
        self.base_amplify_bias = float(np.clip(amplify_bias, 0.0, 0.08))  # ✅ cap lower
        self.jitter_amplify_bias = bool(jitter_amplify_bias)
        self.jitter_range = float(np.clip(jitter_range, 0.0, 0.05))

        # ✅ 현재 에피소드에서 쓰는 amplify_bias (reset 때 결정)
        self.amplify_bias = float(self.base_amplify_bias)

        self.records = read_jsonl(self.log_path)
        if len(self.records) < 5:
            raise RuntimeError(f"Not enough records in {self.log_path}. Need >=5, got {len(self.records)}")

        self.records = self.records[-self.window:]

        # ---- spaces ----
        self.obs_dim = 14
        low = np.zeros((self.obs_dim,), dtype=np.float32)
        high = np.ones((self.obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(int(NUM_ACTIONS))

        # ---- state ----
        self.idx = 0
        self.t = 0

        self.state8 = np.zeros((8,), dtype=np.float32)
        self.state = np.zeros((self.obs_dim,), dtype=np.float32)

        self.dom_hist: Deque[str] = deque(maxlen=self.loop_window)
        self.drift_hist: Deque[float] = deque(maxlen=self.loop_window)

        self.prev_wave: Dict[str, Any] = {}
        self.curr_wave: Dict[str, Any] = {}

        # ✅ action history (recent_action_ids 용)
        self.action_hist: Deque[int] = deque(maxlen=self.diversity_k)

    # ------------------------------------------------------------
    # 내부: obs 만들기
    # ------------------------------------------------------------
    def _build_obs(
        self,
        axes8: np.ndarray,
        drift: float,
        loop_score: float,
        dom_value: float,
        dom_delta: float,
        trend_sign: float,
        rt_gap: float,
    ) -> np.ndarray:
        dom_delta_01 = _signed_to_01(dom_delta)
        trend_sign_01 = _signed_to_01(trend_sign)
        rt_gap_01 = _signed_to_01(rt_gap)

        drift = float(np.clip(drift, 0.0, 1.0))
        loop_score = float(np.clip(loop_score, 0.0, 1.0))
        dom_value = float(np.clip(dom_value, 0.0, 1.0))

        obs = np.concatenate(
            [
                axes8.astype(np.float32, copy=False),
                np.array([drift], dtype=np.float32),
                np.array([loop_score], dtype=np.float32),
                np.array([dom_value], dtype=np.float32),
                np.array([float(dom_delta_01)], dtype=np.float32),
                np.array([float(trend_sign_01)], dtype=np.float32),
                np.array([float(rt_gap_01)], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        obs = _to_f32(obs).reshape(self.observation_space.shape)
        obs = _clip_box(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _build_wave_dict(
        self,
        axes8: np.ndarray,
        drift: float,
        loop_score: float,
        dom_axis: str,
        dom_delta: float,
    ) -> Dict[str, Any]:
        axes8 = np.asarray(axes8, dtype=np.float32).reshape((8,))
        T = float(axes8[6])
        R = float(axes8[7])

        pace = pace_from_TR(T, R)
        kind = loop_kind_from(dom_axis, float(loop_score), float(drift))

        if dom_delta > 0.05:
            trend = "up"
        elif dom_delta < -0.05:
            trend = "down"
        else:
            trend = "flat"

        session = {"t": int(self.t), "t_max": int(self.episode_len)}

        # ✅ diversity 보너스 설정
        if self.enable_diversity_bonus:
            session["enable_diversity_bonus"] = True
            session["diversity_k"] = int(self.diversity_k)
            session["diversity_alpha"] = float(self.diversity_alpha)

        # ✅ AMPLIFY 선호도 (reset에서 정한 값)
        session["amplify_bias"] = float(self.amplify_bias)

        return {
            "pace": pace,
            "drift": float(drift),
            "trend": trend,
            "loop": {"score": float(loop_score), "kind": kind},
            "session": session,
            "delta": {dom_axis: float(dom_delta)},
        }

    # ------------------------------------------------------------
    # reset
    # ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.idx = random.randint(0, max(0, len(self.records) - 2))
        self.t = 0

        # ✅ [FIX] 에피소드마다 amplify_bias를 "낮게 + 약간 흔들기"
        if self.jitter_amplify_bias:
            jit = (random.random() * 2.0 - 1.0) * self.jitter_range  # [-range, +range]
            self.amplify_bias = float(np.clip(self.base_amplify_bias + jit, 0.0, 0.08))
        else:
            self.amplify_bias = float(self.base_amplify_bias)

        axes = axes_from_record(self.records[self.idx])
        axes8 = np.array([axes[k] for k in KEYS], dtype=np.float32)
        self.state8 = axes8

        self.dom_hist.clear()
        self.drift_hist.clear()
        self.dom_hist.append(dominant_axis_from_state(self.state8))
        self.drift_hist.append(0.0)

        self.action_hist.clear()

        drift = 0.0
        loop_score = 0.0
        dom_axis, dom_value, dom_delta = dom_value_and_delta(self.state8, self.state8)
        trend_sign = 0.0
        rt_gap = float(self.state8[7] - self.state8[6])

        self.state = self._build_obs(self.state8, drift, loop_score, dom_value, dom_delta, trend_sign, rt_gap)

        self.prev_wave = self._build_wave_dict(self.state8, drift, loop_score, dom_axis, dom_delta)
        self.curr_wave = dict(self.prev_wave)

        if isinstance(self.curr_wave.get("session", {}), dict):
            self.curr_wave["session"]["recent_action_ids"] = list(self.action_hist)

        info = {
            "idx": self.idx,
            "t": self.t,
            "dom_axis": dom_axis,
            "dom_value": dom_value,
            "dom_delta": dom_delta,
            "trend_sign": trend_sign,
            "rt_gap": rt_gap,
            "drift": drift,
            "loop_score": loop_score,
            "amplify_bias": float(self.amplify_bias),
        }
        return self.state, info

    # ------------------------------------------------------------
    # step
    # ------------------------------------------------------------
    def step(self, action: int):
        prev8 = self.state8.copy()

        action_id = int(action)
        goal, style = action_pair(action_id)

        self.idx = min(self.idx + 1, len(self.records) - 1)
        self.t += 1

        axes = axes_from_record(self.records[self.idx])
        cur8 = np.array([axes[k] for k in KEYS], dtype=np.float32)
        self.state8 = cur8

        drift = calc_drift(prev8, cur8)
        self.dom_hist.append(dominant_axis_from_state(cur8))
        self.drift_hist.append(drift)
        loop_score = calc_loop_score(self.dom_hist, self.drift_hist)

        dom_axis, dom_value, dom_delta = dom_value_and_delta(prev8, cur8)
        trend_sign = trend_sign_from_dom_delta(dom_delta, th=0.05)
        rt_gap = float(cur8[7] - cur8[6])

        self.prev_wave = dict(self.curr_wave) if isinstance(self.curr_wave, dict) else {}
        self.curr_wave = self._build_wave_dict(cur8, drift, loop_score, dom_axis, dom_delta)

        # ✅ action history 누적 + recent_action_ids 주입
        self.action_hist.append(action_id)
        if isinstance(self.curr_wave.get("session", {}), dict):
            self.curr_wave["session"]["recent_action_ids"] = list(self.action_hist)

        prev_axes_dict = axes8_to_dict(prev8)
        curr_axes_dict = axes8_to_dict(cur8)

        reward = compute_reward(
            prev_axes_in=prev_axes_dict,
            prev_wave_in=self.prev_wave,
            action=(goal, style),
            curr_axes_in=curr_axes_dict,
            curr_wave_in=self.curr_wave,
        )

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.state = self._build_obs(self.state8, drift, loop_score, dom_value, dom_delta, trend_sign, rt_gap)

        terminated = (self.idx >= len(self.records) - 1) or (self.t >= self.episode_len)
        truncated = False

        info = {
            "idx": self.idx,
            "t": self.t,
            "goal": goal,
            "style": style,
            "action": int(action_id),
            "dom_axis": dom_axis,
            "dom_value": dom_value,
            "dom_delta": dom_delta,
            "trend_sign": trend_sign,
            "rt_gap": rt_gap,
            "drift": float(drift),
            "loop_score": float(loop_score),
            "loop_kind": str(self.curr_wave.get("loop", {}).get("kind", "none")) if isinstance(self.curr_wave, dict) else "none",
            "pace": str(self.curr_wave.get("pace", "mid")) if isinstance(self.curr_wave, dict) else "mid",
            "trend": str(self.curr_wave.get("trend", "flat")) if isinstance(self.curr_wave, dict) else "flat",
            "recent_action_ids": list(self.action_hist),
            "enable_diversity_bonus": bool(self.enable_diversity_bonus),
            "diversity_k": int(self.diversity_k),
            "diversity_alpha": float(self.diversity_alpha),
            "amplify_bias": float(self.amplify_bias),
        }
        return self.state, float(reward), terminated, truncated, info

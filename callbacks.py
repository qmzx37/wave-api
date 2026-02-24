# offline_env.py
from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from action_space import NUM_ACTIONS, action_pair

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]


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


def clamp01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def axes_from_record(r: dict) -> Dict[str, float]:
    labels = r.get("labels") or r.get("axes") or {}
    out: Dict[str, float] = {}
    for k in KEYS:
        out[k] = clamp01(labels.get(k, 0.0))
    return out


class NoiEOfflineEnv(gym.Env):
    """
    obs: 8축 float32 (F,A,D,J,C,G,T,R)
    action: 12개 (goal×style)
      - goal: CLARIFY/STABILIZE/MOTION/AMPLIFY
      - style: NORMAL/SHORT/DIRECT
    reward: 로그에서 실제 상태 변화량을 기반으로, action이 "맞는 방향"이면 +
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
        except Exception:
            env_config = dict(env_config)
            log_path = env_config.get("log_path")
            window = int(env_config.get("window", 3000))
            episode_len = int(env_config.get("episode_len", 20))

        if not log_path:
            log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")

        self.log_path = str(log_path)
        self.window = max(10, int(window))
        self.episode_len = max(5, int(episode_len))

        self.records = read_jsonl(self.log_path)
        if len(self.records) < 10:
            raise RuntimeError(f"Not enough records in {self.log_path}. Need >=10, got {len(self.records)}")

        self.records = self.records[-self.window:]

        # spaces (float32 고정)
        self.obs_dim = 8
        low = np.zeros((self.obs_dim,), dtype=np.float32)
        high = np.ones((self.obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ✅ 12-action
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.idx = 0
        self.t = 0
        self.state = np.zeros((self.obs_dim,), dtype=np.float32)

        # 에피소드 목표(간단): 랜덤
        self.target = random.choice(["R_UP", "T_DOWN", "J_UP", "D_DOWN"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.idx = random.randint(0, max(0, len(self.records) - (self.episode_len + 2)))
        axes = axes_from_record(self.records[self.idx])

        obs = np.array([axes[k] for k in KEYS], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32, copy=False)

        self.state = obs
        info = {"target": self.target, "idx": self.idx, "t": self.t}
        return self.state, info

    def step(self, action: int):
        prev = self.state.copy()

        # 다음 로그 상태로 이동
        self.idx = min(self.idx + 1, len(self.records) - 1)
        self.t += 1

        axes = axes_from_record(self.records[self.idx])
        obs = np.array([axes[k] for k in KEYS], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32, copy=False)
        self.state = obs

        # 축 인덱스
        F, A, D, J, C, G, T, R = range(8)

        def d(i: int) -> float:
            return float(self.state[i] - prev[i])

        goal, style = action_pair(action)

        # -------------------------
        # (A) goal별 기본 reward
        # -------------------------
        reward = 0.0
        if goal == "STABILIZE":
            reward += 1.0 * d(R) - 1.0 * d(T) - 0.2 * d(F)
        elif goal == "CLARIFY":
            reward += -0.8 * d(A) - 0.5 * d(F) + 0.2 * d(R)
        elif goal == "MOTION":
            reward += -1.0 * d(D) + 0.3 * d(R) + 0.1 * d(J)
        elif goal == "AMPLIFY":
            reward += 1.0 * d(J) + 0.2 * d(C) - 0.2 * d(T)

        # -------------------------
        # (B) style 보정 (말투 비용/효과)
        # - SHORT: 빠르게 핵심 -> 약간 보너스, 하지만 정보 손실 패널티도 조금
        # - DIRECT: 강하게 밀어붙임 -> 불안/분노 높으면 패널티(관계 리스크)
        # -------------------------
        if style == "SHORT":
            reward += 0.05
            reward -= 0.05 * (d(C) < 0.0)  # 탐색이 줄어들면 살짝 패널티
        elif style == "DIRECT":
            # F/A 높을 때 DIRECT는 리스크
            fa = float(prev[F] + prev[A]) / 2.0
            reward -= 0.20 * fa
            reward += 0.05  # 효과 자체는 조금 보너스

        # -------------------------
        # (C) 타겟(에피소드 목표) 보정
        # -------------------------
        if self.target == "R_UP":
            reward += 0.4 * d(R)
        elif self.target == "T_DOWN":
            reward += -0.4 * d(T)
        elif self.target == "J_UP":
            reward += 0.4 * d(J)
        elif self.target == "D_DOWN":
            reward += -0.4 * d(D)

        terminated = (self.t >= self.episode_len) or (self.idx >= len(self.records) - 1)
        truncated = False

        info = {"idx": self.idx, "t": self.t, "target": self.target, "goal": goal, "style": style}
        return self.state, float(reward), terminated, truncated, info

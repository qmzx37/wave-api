# rl_policy_test.py
from __future__ import annotations

import os
import glob
from typing import Any, Dict

import ray
from ray.rllib.algorithms.algorithm import Algorithm

from offline_env import NoiEOfflineEnv
from action_space import action_pair  # action_id -> (goal, style)


def find_latest_checkpoint(root_dir: str) -> str:
    """
    runs 폴더 아래 checkpoint_* 폴더를 찾아서 가장 최근 것을 반환.
    예: C:/llm/train/runs/noie_ppo_offline_12action/**/checkpoint_000000
    """
    pattern = os.path.join(root_dir, "**", "checkpoint_*")
    cands = glob.glob(pattern, recursive=True)
    if not cands:
        raise FileNotFoundError(f"No checkpoint found under: {root_dir}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def main():
    log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")

    # ✅ 1) 체크포인트 경로 결정
    ckpt = os.environ.get("RL_CKPT", "").strip()
    if not ckpt:
        runs_root = os.environ.get("RAY_STORAGE", r"C:\llm\train\runs")
        exp_root = os.path.join(runs_root, "noie_ppo_offline_12action")
        ckpt = find_latest_checkpoint(exp_root)

    ckpt = ckpt.replace("/", "\\")
    print("✅ Using checkpoint:", ckpt)

    # ✅ 2) Ray init (이미 떠있어도 OK)
    ray.init(ignore_reinit_error=True)

    # ✅ 3) 알고리즘 로드
    algo = Algorithm.from_checkpoint(ckpt)

    # ✅ 4) 환경 생성 (학습 때와 동일한 config 추천)
    env = NoiEOfflineEnv({"log_path": log_path, "window": 3000, "episode_len": 20})

    # ✅ 5) 몇 에피소드 돌려서 정책이 고르는 액션 확인
    episodes = int(os.environ.get("EVAL_EPISODES", "5"))
    total_return = 0.0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        step = 0

        print("\n" + "=" * 70)
        print(f"[EP {ep}] target={info.get('target')}")

        while not done:
            # compute_single_action은 obs만 넣으면 됨 (단일 에이전트)
            action = algo.compute_single_action(obs)
            goal, style = action_pair(int(action))

            obs, r, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)

            ep_ret += float(r)
            step += 1

            print(f" step={step:02d} action_id={int(action):02d} -> ({goal},{style})  r={float(r):+.3f}")

        print(f" EP_RETURN = {ep_ret:.3f}")
        total_return += ep_ret

    print("\n✅ AVG_RETURN =", total_return / max(1, episodes))
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

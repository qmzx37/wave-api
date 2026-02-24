from __future__ import annotations

import os
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from offline_env import NoiEOfflineEnv  # ✅ 14축 env여야 함


def main() -> None:
    log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")
    storage_root = os.environ.get("RAY_STORAGE", r"C:\llm\train\runs")
    exp_name = "noie_ppo_offline_12action_obs14"

    window = int(os.environ.get("ENV_WINDOW", "3000"))
    episode_len = int(os.environ.get("EPISODE_LEN", "20"))

    iters = int(os.environ.get("TRAIN_ITERS", "50"))
    lr = float(os.environ.get("LR", "3e-4"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    clip_param = float(os.environ.get("CLIP", "0.2"))

    train_batch_size = int(os.environ.get("TRAIN_BATCH", "4000"))
    sgd_minibatch_size = int(os.environ.get("MINIBATCH", "256"))
    num_sgd_iter = int(os.environ.get("SGD_ITERS", "10"))

    ckpt_every = int(os.environ.get("CKPT_EVERY", "10"))

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    cfg = (
        PPOConfig()
        .environment(
            env=NoiEOfflineEnv,
            env_config={"log_path": log_path, "window": window, "episode_len": episode_len},
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            lr=lr,
            gamma=gamma,
            clip_param=clip_param,
            train_batch_size=train_batch_size,
            minibatch_size=sgd_minibatch_size,
            num_epochs=num_sgd_iter,
        )
    )

    print("=" * 70)
    print("[NoiE PPO OFFLINE TRAIN] obs14")
    print("log_path    =", log_path)
    print("storage_root=", storage_root)
    print("exp_name    =", exp_name)
    print("window      =", window)
    print("episode_len =", episode_len)
    print("iters       =", iters)
    print("train_batch =", train_batch_size)
    print("minibatch   =", sgd_minibatch_size)
    print("sgd_iters   =", num_sgd_iter)
    print("=" * 70)

    # ✅ env obs shape 검증 (여기서 (14,) 아니면 바로 중단)
    env = NoiEOfflineEnv({"log_path": log_path, "window": window, "episode_len": episode_len})
    obs, _ = env.reset()
    print("✅ ENV obs_shape =", getattr(obs, "shape", None), "dtype=", getattr(obs, "dtype", None))
    if getattr(obs, "shape", None) != (14,):
        raise RuntimeError(f"ENV obs is not (14,). Got {getattr(obs,'shape',None)}")

    algo = cfg.build()

    save_root = os.path.join(storage_root, exp_name)
    os.makedirs(save_root, exist_ok=True)

    last_ckpt: str | None = None

    for i in range(1, iters + 1):
        result: dict[str, Any] = algo.train()
        ep_ret = result.get("episode_reward_mean", None)
        ep_len = result.get("episode_len_mean", None)
        timesteps = result.get("timesteps_total", None)

        print(f"[iter {i:03d}] reward_mean={ep_ret} len_mean={ep_len} timesteps={timesteps}")

        if (i % ckpt_every) == 0 or i == iters:
            ckpt_dir = algo.save(checkpoint_dir=save_root)
            last_ckpt = ckpt_dir
            print(f"💾 saved checkpoint -> {ckpt_dir}")

    print("\n✅ DONE")
    if last_ckpt:
        print("✅ LAST_CKPT =", last_ckpt)
        print("👉 PowerShell:")
        print(f'$env:RL_CKPT="{last_ckpt}"')

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

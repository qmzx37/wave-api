# train_rllib_ppo.py  (obs14 + 12-action + diversity + amplify_bias 옵션 포함)
from __future__ import annotations

import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from offline_env import NoiEOfflineEnv


def env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def main():
    # -------------------------
    # Paths
    # -------------------------
    log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")
    storage_path = os.environ.get("RAY_STORAGE", r"C:\llm\train\runs")
    exp_name = os.environ.get("EXP_NAME", "noie_ppo_offline_12action_obs14_retrain")

    # -------------------------
    # Env settings
    # -------------------------
    window = env_int("ENV_WINDOW", 3000)
    episode_len = env_int("EPISODE_LEN", 20)
    loop_window = env_int("LOOP_WINDOW", 12)
    reward_clip = env_float("REWARD_CLIP", 1.0)

    # diversity bonus (reward.py가 curr_wave.session에서 읽음)
    enable_div = env_bool("ENABLE_DIVERSITY_BONUS", "1")
    diversity_k = env_int("DIVERSITY_K", 12)
    diversity_alpha = env_float("DIVERSITY_ALPHA", 0.04)

    # amplify bias (reward.py가 curr_wave.session["amplify_bias"]로 읽음)
    amplify_bias = env_float("AMPLIFY_BIAS", 0.06)  # ✅ 과몰림 줄이려면 0.00~0.08 권장

    # -------------------------
    # PPO settings
    # -------------------------
    num_gpus = env_int("NUM_GPUS", 0)
    iters = env_int("TRAIN_ITERS", 60)

    gamma = env_float("GAMMA", 0.95)
    lr = env_float("LR", 2e-4)
    train_batch_size = env_int("TRAIN_BATCH_SIZE", 4000)
    minibatch_size = env_int("MINIBATCH_SIZE", 128)
    num_epochs = env_int("NUM_EPOCHS", 10)
    clip_param = env_float("CLIP_PARAM", 0.2)

    # -------------------------
    # Ray init
    # -------------------------
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # -------------------------
    # Env config
    # -------------------------
    env_config = {
        "log_path": log_path,
        "window": window,
        "episode_len": episode_len,
        "loop_window": loop_window,
        "reward_clip": reward_clip,
        "enable_diversity_bonus": bool(enable_div),
        "diversity_k": int(diversity_k),
        "diversity_alpha": float(diversity_alpha),
        "amplify_bias": float(amplify_bias),
    }

    # -------------------------
    # RLlib Config (old stack 고정)
    # -------------------------
    cfg = (
        PPOConfig()
        .environment(env=NoiEOfflineEnv, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0)  # Windows 로컬 안정
        .training(
            gamma=gamma,
            lr=lr,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            clip_param=clip_param,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .resources(num_gpus=num_gpus)
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=cfg.to_dict(),
        run_config=tune.RunConfig(
            name=exp_name,
            storage_path=storage_path,
            stop={"training_iteration": iters},
        ),
    )

    results = tuner.fit()

    # -------------------------
    # Print best checkpoint
    # -------------------------
    try:
        best = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
        print("\n✅ DONE")
        print("storage_path:", storage_path)
        print("experiment name:", exp_name)
        print("best checkpoint:", best.checkpoint.path)
    except Exception as e:
        print("\n⚠️ DONE (but best checkpoint lookup failed):", repr(e))


if __name__ == "__main__":
    main()

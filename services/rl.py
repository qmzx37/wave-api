# services/rl.py
# ✅ obs14 + PPO checkpoint resolve + action selection + anti-collapse
from __future__ import annotations

import os
import glob
import json
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque

import numpy as np

from action_space import action_pair, NUM_ACTIONS


# ============================================================
# env helpers
# ============================================================
def env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def clamp01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


# ============================================================
# ✅ checkpoint auto-discovery
# ============================================================
def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def find_latest_checkpoint(runs_dir: str) -> Optional[str]:
    runs_dir = (runs_dir or "").strip().strip('"').strip("'")
    if not runs_dir or (not os.path.exists(runs_dir)):
        return None

    pattern = os.path.join(runs_dir, "**", "checkpoint_*")
    cands = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    if not cands:
        return None

    cands.sort(key=_safe_mtime, reverse=True)
    return cands[0]


def resolve_rl_ckpt() -> Tuple[Optional[str], Dict[str, Any]]:
    info: Dict[str, Any] = {"source": None, "runs_dir": None, "picked": None, "env": None}

    env_ckpt = os.environ.get("RL_CKPT", "").strip().strip('"').strip("'")
    info["env"] = env_ckpt

    if env_ckpt and os.path.exists(env_ckpt):
        info["source"] = "env"
        info["picked"] = env_ckpt
        return env_ckpt, info

    runs_dir = os.environ.get("PPO_RUNS_DIR", r"C:\llm\train\runs")
    info["runs_dir"] = runs_dir

    latest = find_latest_checkpoint(runs_dir)
    if latest and os.path.exists(latest):
        info["source"] = "auto"
        info["picked"] = latest
        return latest, info

    info["source"] = "none"
    return None, info


# ============================================================
# ✅ obs14
# ============================================================
def axes_to_8vec(axes: Dict[str, float], KEYS: List[str]) -> np.ndarray:
    return np.asarray([float(axes[k]) for k in KEYS], dtype=np.float32)


def _signed_to_01(x: float) -> float:
    return clamp01((float(x) + 1.0) * 0.5)


def calc_drift(prev8: np.ndarray, cur8: np.ndarray) -> float:
    return float(np.mean(np.abs(cur8 - prev8)))


def read_last_axes8_from_log(log_path: str, KEYS: List[str], n: int = 8) -> List[np.ndarray]:
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max(1, n) :]
    except Exception:
        return []

    out: List[np.ndarray] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        labels = rec.get("labels") or rec.get("axes") or {}
        if not isinstance(labels, dict):
            continue
        ax = {k: clamp01(labels.get(k, 0.0)) for k in KEYS}
        out.append(axes_to_8vec(ax, KEYS))
    return out


def calc_loop_score_recent(recent_axes8: List[np.ndarray], KEYS: List[str], k: int = 6) -> float:
    if len(recent_axes8) < k:
        return 0.0
    tail = recent_axes8[-k:]
    doms = [KEYS[int(np.argmax(v))] for v in tail]
    most = max(set(doms), key=doms.count)
    ratio = doms.count(most) / float(k)

    drifts = []
    for i in range(1, k):
        drifts.append(float(np.mean(np.abs(tail[i] - tail[i - 1]))))
    drift_mean = float(np.mean(drifts)) if drifts else 0.0

    low_drift_factor = max(0.0, 0.25 - drift_mean) / 0.25
    score = ratio * (0.5 + 0.5 * low_drift_factor)
    return float(np.clip(score, 0.0, 1.0))


def dom_value_and_delta(prev8: np.ndarray, cur8: np.ndarray, KEYS: List[str]) -> Tuple[str, float, float]:
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


def make_obs14(axes: Dict[str, float], log_path: str, KEYS: List[str]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    cur8 = axes_to_8vec(axes, KEYS)

    recent8 = read_last_axes8_from_log(log_path, KEYS, n=8)
    prev8 = recent8[-2] if len(recent8) >= 2 else cur8

    drift = float(np.clip(calc_drift(prev8, cur8), 0.0, 1.0))

    recent_for_loop = list(recent8)
    if (not recent_for_loop) or (not np.allclose(recent_for_loop[-1], cur8)):
        recent_for_loop.append(cur8)
    loop_score = float(np.clip(calc_loop_score_recent(recent_for_loop, KEYS, k=6), 0.0, 1.0))

    dom_axis, dom_value, dom_delta = dom_value_and_delta(prev8, cur8, KEYS)
    trend_sign = trend_sign_from_dom_delta(dom_delta, th=0.05)
    rt_gap = float(cur8[7] - cur8[6])  # R - T (KEYS 순서가 동일하다는 전제)

    obs = np.concatenate(
        [
            cur8,
            np.array([drift], np.float32),
            np.array([loop_score], np.float32),
            np.array([dom_value], np.float32),
            np.array([_signed_to_01(dom_delta)], np.float32),
            np.array([_signed_to_01(trend_sign)], np.float32),
            np.array([_signed_to_01(rt_gap)], np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    feats = {
        "dom_axis": dom_axis,
        "dom_value": float(dom_value),
        "dom_delta": float(dom_delta),
        "trend_sign": float(trend_sign),
        "rt_gap": float(rt_gap),
        "drift": float(drift),
        "loop_score": float(loop_score),
    }
    return obs.reshape((14,)), loop_score, feats


# ============================================================
# Sampling helpers
# ============================================================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(x) / float(len(x))


def sample_from_logits(logits: np.ndarray, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
    t = max(0.05, float(temperature))
    z = logits.astype(np.float32) / t
    probs = softmax(z)
    a = int(np.random.choice(len(probs), p=probs))
    return a, probs


def topk_pairs(vals: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    k = max(1, min(int(k), int(vals.shape[-1])))
    idx = np.argsort(vals)[::-1][:k]
    out: List[Dict[str, Any]] = []
    for i in idx:
        g, s = action_pair(int(i))
        out.append({"action_id": int(i), "value": float(vals[i]), "goal": g, "style": s})
    return out


# ============================================================
# PPO init + action selection
# ============================================================
LAST_ACTIONS: Deque[int] = deque(maxlen=8)


def init_ppo() -> Tuple[Any, bool, Dict[str, Any]]:
    """
    return: (RL_ALGO, RL_READY, RL_CKPT_INFO)
    """
    RL_CKPT_INFO: Dict[str, Any] = {"source": None, "runs_dir": None, "picked": None, "env": None}

    # old stack 고정
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
    except Exception:
        return None, False, {"source": "none", "error": "ray_or_ppoconfig_missing"}

    try:
        from offline_env import NoiEOfflineEnv
    except Exception:
        return None, False, {"source": "none", "error": "NoiEOfflineEnv_missing"}

    ckpt, ckpt_info = resolve_rl_ckpt()
    RL_CKPT_INFO = dict(ckpt_info or {})

    if not ckpt:
        RL_CKPT_INFO["error"] = "checkpoint_not_found"
        return None, False, RL_CKPT_INFO

    try:
        os.environ["RL_CKPT"] = ckpt
    except Exception:
        pass

    try:
        if (ray is not None) and (not ray.is_initialized()):
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        log_path = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")
        episode_len = int(os.environ.get("EPISODE_LEN", "20"))
        window = int(os.environ.get("ENV_WINDOW", "3000"))

        cfg = (
            PPOConfig()
            .environment(
                env=NoiEOfflineEnv,
                env_config={"log_path": log_path, "window": window, "episode_len": episode_len},
            )
            .framework("torch")
            .env_runners(num_env_runners=0)
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        )

        algo = cfg.build()
        algo.restore(ckpt)
        RL_CKPT_INFO["restored"] = True
        return algo, True, RL_CKPT_INFO

    except Exception as e:
        RL_CKPT_INFO["error"] = repr(e)
        return None, False, RL_CKPT_INFO


def select_action(
    *,
    RL_READY: bool,
    RL_ALGO: Any,
    obs14: np.ndarray,
) -> Tuple[int, Dict[str, Any]]:
    """
    return: (action_id, debug_rl)
    """
    force_rlmodule = env_bool("PPO_DEBUG_FORCE_RLMODULE", "0")
    sample_from_logits_flag = env_bool("PPO_SAMPLE_FROM_LOGITS", "1")
    eps_greedy = float(np.clip(env_float("PPO_EPS_GREEDY", 0.25), 0.0, 1.0))
    temperature = float(np.clip(env_float("PPO_TEMPERATURE", 1.6), 0.05, 5.0))
    topk = env_int("PPO_TOPK", 5)

    anti_collapse = env_bool("PPO_ANTI_COLLAPSE", "1")
    anti_k = env_int("PPO_ANTI_K", 6)
    anti_eps = float(np.clip(env_float("PPO_ANTI_EPS", 0.60), 0.0, 1.0))

    debug_rl: Dict[str, Any] = {
        "rl_ready": bool(RL_READY),
        "used_ppo": False,
        "error": None,
        "mode": None,
        "obs14_shape": list(obs14.shape),
        "eps_greedy": float(eps_greedy),
        "temperature": float(temperature),
        "sample_from_logits": bool(sample_from_logits_flag),
        "force_rlmodule": bool(force_rlmodule),
        "anti_collapse": bool(anti_collapse),
        "anti_k": int(anti_k),
        "anti_eps": float(anti_eps),
    }

    action_id = 0
    if RL_READY and RL_ALGO is not None:
        try:
            if np.random.rand() < eps_greedy:
                action_id = int(np.random.randint(0, int(NUM_ACTIONS)))
                debug_rl["used_ppo"] = True
                debug_rl["mode"] = "eps_greedy_random"
            else:
                logits = None

                if force_rlmodule or sample_from_logits_flag:
                    try:
                        m = RL_ALGO.get_module()
                        batch_np = {"obs": np.expand_dims(obs14, axis=0).astype(np.float32, copy=False)}
                        out = m.forward_inference(batch_np)

                        if isinstance(out, dict) and "action_dist_inputs" in out:
                            log = np.array(out["action_dist_inputs"])
                            logits = log[0] if log.ndim == 2 else log.reshape(-1)
                            debug_rl["mode"] = "rlmodule_logits"
                            debug_rl["used_ppo"] = True
                        else:
                            raise KeyError("forward_inference missing action_dist_inputs")
                    except Exception as e_mod:
                        debug_rl["rlmodule_error"] = repr(e_mod)
                        logits = None

                if logits is not None:
                    debug_rl["top5_logits"] = topk_pairs(logits, k=topk)

                    if sample_from_logits_flag:
                        action_id, probs = sample_from_logits(logits, temperature=temperature)
                        debug_rl["mode"] = str(debug_rl["mode"]) + "+sample"
                        debug_rl["top5_probs"] = topk_pairs(probs, k=topk)
                    else:
                        action_id = int(np.argmax(logits))
                        debug_rl["mode"] = str(debug_rl["mode"]) + "+argmax"
                else:
                    out = RL_ALGO.compute_single_action(obs14, explore=True)
                    action_id = int(out) if not isinstance(out, (tuple, list)) else int(out[0])
                    debug_rl["used_ppo"] = True
                    debug_rl["mode"] = "compute_single_action(explore=True)"

            if action_id < 0 or action_id >= int(NUM_ACTIONS):
                action_id = 0

        except Exception as e:
            debug_rl["error"] = repr(e)
            action_id = 0

    # anti-collapse
    try:
        LAST_ACTIONS.append(int(action_id))
        if anti_collapse and len(LAST_ACTIONS) >= int(anti_k):
            tail = list(LAST_ACTIONS)[-int(anti_k) :]
            if len(set(tail)) == 1 and np.random.rand() < anti_eps:
                old = int(action_id)
                candidates = [i for i in range(int(NUM_ACTIONS)) if i != old]
                action_id = int(np.random.choice(candidates))
                debug_rl["mode"] = str(debug_rl.get("mode")) + "+anti_collapse"
                debug_rl["anti_from"] = old
                debug_rl["anti_to"] = int(action_id)
    except Exception:
        pass

    # attach action_pair
    g, s = action_pair(int(action_id))
    debug_rl["action_pair"] = {"goal": g, "style": s}
    debug_rl["action_id"] = int(action_id)

    return int(action_id), debug_rl

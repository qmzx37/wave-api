# C:\llm\train\infer_lora_8axis_final.py
# ------------------------------------------------------------
# 목적:
# - LoRA adapter 로드(4bit QLoRA 환경에서도 안정)
# - 생성 출력에서 JSON만 "관대하게 추출/복구"
# - 키 강제(F,A,D,J,C,G,T,R) + float 변환 + 0~1 clamp
# - 실패 시 재시도(최대 N회)
# - (옵션) 결과를 JSONL 로그로 append 저장
# - (NEW) 파동(Ψ) 상태 유지 + LoRA OFF 상태로 "자유 대화 답변" 생성
# - (PATCH) wave_state 안의 action을 meta.action으로 보존 저장
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import time
import random
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]

# -------------------------
# Utils
# -------------------------
def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _env_bool(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _strip_code_fence(s: str) -> str:
    return s.replace("```json", "").replace("```", "").strip()


def hidden_d_risk(axes: dict) -> bool:
    T = float(axes.get("T", 0.0))
    C = float(axes.get("C", 0.0))
    D = float(axes.get("D", 0.0))
    J = float(axes.get("J", 0.0))
    R = float(axes.get("R", 0.0))
    # 압박+산만이 큰데 우울/기쁨은 낮고 안정도 낮으면 "숨은 회피/무기력" 가능성
    return (T >= 0.60 and C >= 0.25 and D <= 0.20 and J <= 0.30 and R <= 0.40)


def probe_question_for_focus(axes: dict) -> str:
    T = float(axes.get("T", 0.0))
    C = float(axes.get("C", 0.0))

    probes = [
        "딴 생각이 재밌어서 그래, 아니면 불안해서 도망치는 느낌이야?",
        "지금 더 큰 건 압박(T)이야, 아니면 무기력(D)이야?",
        "딴 생각 들 때 자책이 같이 와? 아니면 그냥 산만한 거야?"
    ]
    if T >= 0.70:
        return probes[0]
    if C >= 0.50:
        return probes[2]
    return random.choice(probes)


def _repair_json_like(blob: str) -> Optional[dict]:
    """
    JSON이 살짝 깨져서 나오는 경우를 복구 시도:
    - 끝의 '}' 누락
    - 끝의 쉼표
    - 잘린 출력(대부분 마지막 괄호가 빠짐)
    """
    if not blob:
        return None

    t = blob.strip()

    # 가장 흔한 케이스: 마지막 } 누락
    if t.startswith("{") and not t.endswith("}"):
        t2 = t + "}"
    else:
        t2 = t

    # trailing comma 제거
    t2 = re.sub(r",\s*}", "}", t2)

    # 마지막 } 이후는 제거
    if "}" in t2:
        t2 = t2[: t2.rfind("}") + 1]

    try:
        return json.loads(t2)
    except Exception:
        return None


def extract_axes_json(raw_text: str) -> Optional[dict]:
    """
    출력 텍스트에서 JSON 블록을 관대하게 추출.
    - 정상 JSON 블록이 있으면 마지막 블록부터 시도
    - 깨진 JSON은 repair 시도
    - 키 8개(F,A,D,J,C,G,T,R) 강제
    """
    if not raw_text:
        return None

    cleaned = _strip_code_fence(raw_text)

    # 1) 정상적으로 닫힌 {...} 블록들
    matches = re.findall(r"\{[\s\S]*?\}", cleaned)
    candidates = list(reversed(matches))

    # 2) '{ ...'로 시작만 하고 '}'가 잘렸을 때
    if "{" in cleaned and "}" not in cleaned:
        candidates.insert(0, cleaned[cleaned.find("{") :])

    # 3) 마지막 '{'부터 끝까지
    if "{" in cleaned and "}" in cleaned:
        tail = cleaned[cleaned.rfind("{") :]
        candidates.insert(0, tail)

    for blob in candidates:
        obj = None
        try:
            obj = json.loads(blob)
        except Exception:
            obj = _repair_json_like(blob)

        if not isinstance(obj, dict):
            continue

        out = {}
        ok = True
        for k in KEYS:
            if k not in obj:
                ok = False
                break
            out[k] = clamp01(obj.get(k, 0.0))

        if ok:
            return out

    return None


def build_axes_messages(user_text: str) -> List[dict]:
    """
    LLaMA3 chat template용 messages 구성 (축 추출용).
    """
    schema = """{
  "F": 0.0,
  "A": 0.0,
  "D": 0.0,
  "J": 0.0,
  "C": 0.0,
  "G": 0.0,
  "T": 0.0,
  "R": 0.0
}"""
    system = (
        "너는 한국어 감정 분석기다. 반드시 JSON 하나만 출력한다.\n"
        "절대 설명/문장/코드블록/추가키를 출력하지 마라.\n"
        "키는 F,A,D,J,C,G,T,R 8개만 허용한다.\n"
        "각 값은 0~1 사이 number(숫자)여야 한다."
    )
    user = (
        f"문장: {user_text}\n\n"
        "반드시 아래 JSON 스키마 형태로만 출력해. 다른 텍스트 금지.\n"
        f"{schema}\n"
        "각 값은 0~1 사이 number(숫자)여야 한다."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -------------------------
# (NEW v2) Wave Engine (Ψ) : 2nd-order approximation
# -------------------------
def init_wave_state() -> Dict[str, float]:
    state = {f"psi_{k}": 0.0 for k in KEYS}
    state.update({f"dpsi_{k}": 0.0 for k in KEYS})

    now = time.time()
    state["session_t0"] = now
    state["last_t"] = now

    state["omega"] = 3.0   # rad/s
    state["zeta"] = 0.2
    state["freq_hz"] = state["omega"] / (2.0 * 3.141592)

    return state


def update_wave_state(state: Dict[str, float], axes: Dict[str, float], now: Optional[float] = None) -> Dict[str, float]:
    """
    2차(감쇠 진동) 근사:
      x'' + 2ζω x' + ω^2 x = ω^2 u
    여기서 u는 axes[k], x는 psi_k
    """
    now = now or time.time()
    dt = max(0.01, min(0.25, now - float(state.get("last_t", now))))
    state["last_t"] = now

    T = clamp01(axes.get("T", 0.0))
    R = clamp01(axes.get("R", 0.0))

    omega = 1.3 + 12.0 * T
    zeta = clamp01(0.02 + 1.02 * (R ** 1.6))

    state["omega"] = float(omega)
    state["zeta"] = float(zeta)
    state["freq_hz"] = float(omega / (2.0 * 3.141592))

    v_abs_sum = 0.0

    for k in KEYS:
        x = float(state.get(f"psi_{k}", 0.0))
        v = float(state.get(f"dpsi_{k}", 0.0))
        u = clamp01(axes.get(k, 0.0))

        a = (omega * omega) * (u - x) - 2.0 * zeta * omega * v

        v = v + a * dt
        v = max(-3.0, min(3.0, v))

        x = x + v * dt
        x = clamp01(x)

        state[f"dpsi_{k}"] = float(v)
        state[f"psi_{k}"] = float(x)

        v_abs_sum += abs(v)

    # 평균 속도(디버그)
    state["v"] = float(v_abs_sum / float(len(KEYS)))

    elapsed = now - float(state.get("session_t0", now))
    if elapsed >= 30 * 60:
        bump = clamp01((elapsed - 30 * 60) / (10 * 60)) * 0.08
        state["psi_R"] = clamp01(state.get("psi_R", 0.0) + bump)
        state["psi_T"] = clamp01(state.get("psi_T", 0.0) - bump * 0.8)

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

    dom = max(["F", "A", "D", "J"], key=lambda k: float(state.get(f"psi_{k}", 0.0)))
    dom_val = float(state.get(f"psi_{dom}", 0.0))

    omega = float(state.get("omega", 6.0))
    zeta  = float(state.get("zeta", 0.25))
    vbar  = float(state.get("v", 0.0))
    hz = omega / (2.0 * math.pi)

    return {
        "pace": pace,
        "dominant": dom,
        "dominant_val": dom_val,
        "psi_T": psiT,
        "psi_R": psiR,
        "omega": omega,
        "hz": hz,
        "zeta": zeta,
        "v": vbar,
    }


# -------------------------
# (NEW) Response Builder (자유 대화)
# -------------------------
def choose_opener(axes: Dict[str, float], wave_summary: Dict[str, float | str]) -> str:
    pace = str(wave_summary.get("pace", "mid"))
    dominant = str(wave_summary.get("dominant", "J"))

    by_dom = {
        "J": ["오", "좋네", "ㅋㅋ", "음"],
        "A": ["아", "하", "음", "야"],
        "D": ["음", "하…", "아", "그래"],
        "F": ["음", "아", "그래", "잠깐"],
    }
    cand = by_dom.get(dominant, ["음", "그래", "오", "야"])

    if pace == "slow":
        for x in ["음", "하…", "그래", "잠깐"]:
            if x in cand:
                return x
        return "음"
    if pace == "fast":
        for x in ["오", "야", "아", "좋네"]:
            if x in cand:
                return x
        return "오"
    return cand[0]


def build_chat_messages(user_text: str, axes: Dict[str, float], wave_summary: Dict[str, float | str]) -> List[dict]:
    dom = str(wave_summary.get("dominant", "J"))
    pace = str(wave_summary.get("pace", "mid"))
    psiT = float(wave_summary.get("psi_T", 0.0))
    psiR = float(wave_summary.get("psi_R", 0.0))

    C = float(axes.get("C", 0.0))
    G = float(axes.get("G", 0.0))

    opener = choose_opener(axes, wave_summary)

    system = (
        "너는 '노이에'다. 감정 파동 기반 대화 AI다.\n"
        "목표: 사용자의 말을 과하게 해석하지 말고, 짧게 미러링하며 대화를 이어간다.\n"
        "규칙:\n"
        "- 한국어로 2~4문장.\n"
        "- 기본은 반말(존댓말/요체 금지).\n"
        "- 판단/설교/훈계/상담사 말투 금지.\n"
        "- 사용자가 요청하지 않으면 해결책을 쏟아내지 않는다.\n"
        "- C(호기심)나 G(욕심)가 높을 때만: '장소' 또는 '대화거리(영화/넷플릭스)'를 1개 정도 가볍게 제안.\n"
        "- T(긴장)가 높으면 문장을 더 짧게, 속도를 낮추는 표현(숨고르기/정리) 사용.\n"
        "- R(안정)이 높으면 자연스럽게 마무리/종료 쪽(오늘은 여기까지도 괜찮다)을 아주 약하게 섞는다.\n"
        "- 절대 JSON을 출력하지 마라.\n"
        "- '지치네' 같은 고정 멘트로 시작하지 마라.\n"
    )

    hint = f"(파동 힌트: dom={dom}, pace={pace}, T≈{psiT:.2f}, R≈{psiR:.2f}, C≈{C:.2f}, G≈{G:.2f})"
    user = (
        f"사용자 말: {user_text}\n"
        f"{hint}\n"
        f"답변은 반드시 '{opener}'로 시작해.\n"
        "위 규칙대로 노이에로서 답변해."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -------------------------
# Model Load
# -------------------------
def load_model():
    BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    ADAPTER = os.environ.get("ADAPTER", r"C:\llm\lora_8axis_adapter_v2")
    OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", r"C:\llm\offload")

    print(f"📦 Base model: {BASE_MODEL}")
    print(f"🧩 LoRA path: {ADAPTER}")
    print(f"🧯 Offload dir: {OFFLOAD_DIR}")

    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_memory = {
        0: os.environ.get("GPU_MAX_MEM", "7GiB"),
        "cpu": os.environ.get("CPU_MAX_MEM", "32GiB"),
    }

    model_kwargs = dict(
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        offload_folder=OFFLOAD_DIR,
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model.eval()

    lora_ok = True
    try:
        model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
    except Exception as e:
        lora_ok = False
        print("⚠️ LoRA load failed, using base model only.")
        print("   reason:", repr(e))
        model.eval()

    return tok, model, lora_ok


# -------------------------
# Inference: Axes(JSON) Extraction (LoRA ON)
# -------------------------
def infer_once(tok, model, user_text: str) -> Tuple[str, Optional[dict], str]:
    MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
    GEN_TEMP = float(os.environ.get("GEN_TEMP", os.environ.get("TEMP_OVERRIDE", "0.2")))
    TOP_P = float(os.environ.get("TOP_P", "0.9"))

    show_debug = _env_bool("SHOW_DEBUG", "0")

    messages = build_axes_messages(user_text)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=GEN_TEMP,
            top_p=TOP_P,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if show_debug:
        print("\n[GEN TEXT]\n", text[:300], "\n")
        print("[GEN TAIL]\n", text[-300:], "\n")

    axes = extract_axes_json(text)

    if axes is None:
        if show_debug:
            print("\n[DEBUG] extract_axes_json failed. RAW GENERATED OUTPUT:\n", text, "\n")
        return text, None, ""

    safe = json.dumps(axes, ensure_ascii=False)
    return text, axes, safe


def infer_with_retries(tok, model, user_text: str) -> dict:
    RETRIES = int(os.environ.get("RETRIES", "2"))

    last_debug = None
    for attempt in range(RETRIES + 1):
        raw_text, axes, safe = infer_once(tok, model, user_text)
        if axes is not None and all(k in axes for k in KEYS):
            return {
                "ok": True,
                "axes": axes,
                "debug": {"attempt": attempt, "safe": safe},
            }
        last_debug = {"attempt": attempt, "safe": safe, "raw_tail": raw_text[-400:] if raw_text else ""}

    return {
        "ok": False,
        "axes": {k: 0.0 for k in KEYS},
        "debug": last_debug or {},
    }


# -------------------------
# (NEW) Chat Generation (LoRA OFF if possible)
# -------------------------
def chat_once(tok, model, user_text: str, axes: dict, wave_state: dict, lora_ok: bool) -> str:
    MAX_NEW = int(os.environ.get("CHAT_MAX_NEW", "160"))
    GEN_TEMP = float(os.environ.get("CHAT_TEMP", "0.7"))
    TOP_P = float(os.environ.get("CHAT_TOP_P", "0.9"))

    ws = summarize_wave(wave_state)
    messages = build_chat_messages(user_text, axes, ws)

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    ctx = nullcontext()
    if lora_ok and hasattr(model, "disable_adapter"):
        try:
            ctx = model.disable_adapter()
        except Exception:
            ctx = nullcontext()

    with ctx:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=GEN_TEMP,
                top_p=TOP_P,
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
            )

    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    if len(text) > 800:
        text = text[:800].rstrip()

    return text


# -------------------------
# Logging
# -------------------------
# -------------------------
# Logging (v2: action 강제 포함 + 하위호환)
# -------------------------
def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _safe_str(x, default="") -> str:
    try:
        s = str(x)
        return s
    except Exception:
        return str(default)

def _extract_action_from_wave_state(wave_state: dict) -> dict:
    """
    wave_state 안에서 action 정보를 최대한 찾아서 표준 형태로 통일.
    허용 입력 예:
      wave_state["action"] = {"id": 3, "goal": "AMPLIFY", "style": "NORMAL"}
      wave_state["meta"]["action"] = ...
      wave_state["wave"]["action"] = ...
    """
    if not isinstance(wave_state, dict):
        return {"id": 0, "goal": "CLARIFY", "style": "NORMAL"}

    # 1) wave_state["action"]
    a = wave_state.get("action")
    if isinstance(a, dict):
        return {
            "id": _safe_int(a.get("id", 0), 0),
            "goal": _safe_str(a.get("goal", "CLARIFY"), "CLARIFY"),
            "style": _safe_str(a.get("style", "NORMAL"), "NORMAL"),
        }

    # 2) wave_state["wave"]["action"]
    w = wave_state.get("wave")
    if isinstance(w, dict) and isinstance(w.get("action"), dict):
        a = w.get("action", {})
        return {
            "id": _safe_int(a.get("id", 0), 0),
            "goal": _safe_str(a.get("goal", "CLARIFY"), "CLARIFY"),
            "style": _safe_str(a.get("style", "NORMAL"), "NORMAL"),
        }

    # 3) wave_state["meta"]["action"]
    m = wave_state.get("meta")
    if isinstance(m, dict) and isinstance(m.get("action"), dict):
        a = m.get("action", {})
        return {
            "id": _safe_int(a.get("id", 0), 0),
            "goal": _safe_str(a.get("goal", "CLARIFY"), "CLARIFY"),
            "style": _safe_str(a.get("style", "NORMAL"), "NORMAL"),
        }

    return {"id": 0, "goal": "CLARIFY", "style": "NORMAL"}


def append_log(path: str, user_text: str, axes: dict, ok: bool, reply: str, wave_state: dict):
    ws = summarize_wave(wave_state)

    # ✅ action은 wave_state에 넣어둔 걸 우선 사용
    action = None
    if isinstance(wave_state, dict):
        action = wave_state.get("action")

    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": user_text,
        "labels": axes,
        "meta": {
            "ok": ok,
            "synthetic": False,
            "wave": ws,
            "action": action,      # ✅ meta.action
            "wave_state": wave_state,  # ✅ meta.wave_state (전체)
        },
        "reply": reply,
    }

    # ✅ (옵션) meta.wave.action 도 같이 박고 싶으면:
    if isinstance(action, dict) and isinstance(rec["meta"].get("wave"), dict):
        rec["meta"]["wave"]["action"] = action

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------------------
# Main (CLI loop)
# -------------------------
def main():
    LOG_PATH = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")
    show_axes = _env_bool("SHOW_AXES", "1")

    print("✅ Loading model...")
    tok, model, lora_ok = load_model()
    print("✅ Ready.")
    print("- 입력: 문장")
    print("- 종료: q")

    wave_state = init_wave_state()

    while True:
        user_text = input("\n문장> ").strip()
        if not user_text:
            continue
        if user_text.lower() == "q":
            break

        t0 = time.time()

        # 1) 축 추출 (LoRA ON)
        res = infer_with_retries(tok, model, user_text)
        axes = res["axes"]

        # 2) 파동 업데이트 (Ψ)
        update_wave_state(wave_state, axes)

        # 3) 대화 답변 (LoRA OFF)
        reply = chat_once(tok, model, user_text, axes, wave_state, lora_ok)

        # 숨은 우울/회피/압박 가능성 탐침 질문 1줄만 덧붙이기
        if hidden_d_risk(axes):
            reply = reply.rstrip() + "\n" + probe_question_for_focus(axes)

        dt = time.time() - t0

        if show_axes:
            print(json.dumps(axes, ensure_ascii=False))
            print(f"   (ok={res['ok']}  time={dt:.2f}s  attempt={res['debug'].get('attempt')})")
        print("\n노이에>", reply)

        # 로그 저장
        append_log(LOG_PATH, user_text, axes, res["ok"], reply, wave_state)
        print(f"\n   saved -> {LOG_PATH}")

    print("\n👋 종료.")


if __name__ == "__main__":
    main()

# wave_ab_test.py
# ------------------------------------------------------------
# 목적:
# - 같은 입력에 대해 FAST vs SLOW 파동을 명확히 분리
# - LLM 자유생성 제거
# - response_builder.policy_v2_friend + wave_summary 기반
#   "친구 같은 채팅" A/B 테스트
#
# 실행:
#   PS> cd C:\llm\train\wave
#   PS> python .\wave_ab_test.py
# ------------------------------------------------------------

from __future__ import annotations

import math
import re
from typing import Dict, List, Any

from infer_lora_8axis_final import (
    load_model,
    infer_with_retries,
    init_wave_state,
    update_wave_state,
)

from response_builder import policy_v2_friend


# -------------------------
# Utils
# -------------------------
def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def force_wave_from_axes(axes: Dict[str, float], psi_T: float, psi_R: float) -> Dict[str, float]:
    """
    axes → wave_state 생성 후
    psi_T / psi_R 강제 주입 (A/B 실험용)
    """
    st = init_wave_state()
    update_wave_state(st, axes)

    st["psi_T"] = clamp01(psi_T)
    st["psi_R"] = clamp01(psi_R)

    return st


def wave_metrics(wave_state: Dict[str, float], axes: Dict[str, float]) -> Dict[str, Any]:
    psi_T = clamp01(wave_state.get("psi_T", 0.0))
    psi_R = clamp01(wave_state.get("psi_R", 0.0))

    omega = 1.3 + 12.0 * psi_T
    hz = omega / (2.0 * math.pi)
    zeta = clamp01(0.02 + 1.02 * (psi_R ** 1.6))

    C = clamp01(axes.get("C", 0.0))
    G = clamp01(axes.get("G", 0.0))
    T = clamp01(axes.get("T", 0.0))

    v = clamp01(
        0.02
        + 0.06 * psi_T
        - 0.04 * psi_R
        + 0.10 * (C + G) / 2.0
        + 0.05 * T
    )

    return {"omega": omega, "hz": hz, "zeta": zeta, "v": v}


def wave_dom(wave_state: Dict[str, float]) -> Dict[str, Any]:
    cand = ["F", "A", "D", "J"]
    vals = {k: float(wave_state.get(f"psi_{k}", 0.0)) for k in cand}
    dom = max(cand, key=lambda k: vals[k])
    return {"dominant": dom, "dominant_val": vals[dom]}


def analyze_reply(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    char_len = len(t)
    sent_cnt = len([s for s in re.split(r"[\.!\?]\s*|\n+", t) if s.strip()])
    honorific = bool(re.search(r"(요$|습니다|세요|해요)", t))
    return {
        "chars": char_len,
        "sents": sent_cnt,
        "honorific": honorific,
    }


def clean_reply(text: str) -> str:
    """
    - '노이에>' 중복 제거
    - 불필요한 공백 정리
    """
    t = re.sub(r"^\s*노이에>\s*", "", text or "")
    return t.strip()


# -------------------------
# Main
# -------------------------
def main():
    print("✅ Loading model...")
    tok, model, _ = load_model()
    print("✅ Model ready.\n")

    tests: List[str] = [
        "나 공부해야하는데 자꾸 딴 생각이 들어",
        "오늘은 기분이 너무 좋아서 뭔가 하고 싶어",
        "괜히 불안해서 집중이 안 돼",
        "짜증이 확 나는데 이유를 모르겠어",
        "그냥 좀 지치고 아무것도 하기 싫다",
    ]

    A = {"name": "A_FAST", "psi_T": 0.90, "psi_R": 0.10, "pace": "high"}
    B = {"name": "B_SLOW", "psi_T": 0.10, "psi_R": 0.90, "pace": "low"}

    for i, user_text in enumerate(tests, 1):
        print("=" * 86)
        print(f"[TEST {i}] 입력: {user_text}")

        res = infer_with_retries(tok, model, user_text)
        axes = res["axes"]

        print(
            f"- axes ok={res['ok']}  F/A/D/J/C/G/T/R="
            f"{axes['F']:.2f}/{axes['A']:.2f}/{axes['D']:.2f}/{axes['J']:.2f}/"
            f"{axes['C']:.2f}/{axes['G']:.2f}/{axes['T']:.2f}/{axes['R']:.2f}"
        )

        # wave state
        wave_A = force_wave_from_axes(axes, A["psi_T"], A["psi_R"])
        wave_B = force_wave_from_axes(axes, B["psi_T"], B["psi_R"])

        # 🔥 핵심: A/B 각각 policy를 따로 호출
        reply_A = policy_v2_friend(
            user_text=user_text,
            axes=axes,
            wave_summary={"pace": A["pace"]},
        )
        reply_B = policy_v2_friend(
            user_text=user_text,
            axes=axes,
            wave_summary={"pace": B["pace"]},
        )

        reply_A = clean_reply(reply_A)
        reply_B = clean_reply(reply_B)

        domA = wave_dom(wave_A)
        domB = wave_dom(wave_B)

        mA = wave_metrics(wave_A, axes)
        mB = wave_metrics(wave_B, axes)

        aA = analyze_reply(reply_A)
        aB = analyze_reply(reply_B)

        print()
        print(
            f"[{A['name']}] pace=fast dom={domA['dominant']}({domA['dominant_val']:.2f}) "
            f"T={A['psi_T']:.2f} R={A['psi_R']:.2f} "
            f"ω={mA['omega']:.2f}rad/s Hz={mA['hz']:.2f} ζ={mA['zeta']:.2f} v={mA['v']:.2f} | "
            f"chars={aA['chars']} sents={aA['sents']} honorific={aA['honorific']}"
        )
        print("노이에>", reply_A)

        print()
        print(
            f"[{B['name']}] pace=slow dom={domB['dominant']}({domB['dominant_val']:.2f}) "
            f"T={B['psi_T']:.2f} R={B['psi_R']:.2f} "
            f"ω={mB['omega']:.2f}rad/s Hz={mB['hz']:.2f} ζ={mB['zeta']:.2f} v={mB['v']:.2f} | "
            f"chars={aB['chars']} sents={aB['sents']} honorific={aB['honorific']}"
        )
        print("노이에>", reply_B)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

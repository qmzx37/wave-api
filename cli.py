from __future__ import annotations

import os
import json
import time

from infer_lora_8axis_final import load_model, infer_with_retries, append_log, KEYS
from response_builder import policy_v2_friend
from state import load_wave_summary


def main():
    LOG_PATH = os.environ.get("LOG_PATH", r"C:\llm\train\emotion_log.jsonl")

    print("✅ Loading model...")
    tok, model = load_model()
    print("✅ Ready.")
    print("- 입력: 문장")
    print("- 종료: 빈 입력 또는 q")

    while True:
        user_text = input("\n문장> ").strip()
        if not user_text or user_text.lower() == "q":
            break

        t0 = time.time()
        res = infer_with_retries(tok, model, user_text)
        dt = time.time() - t0

        axes = res.get("axes", {})
        # 출력: JSON은 필요하면 남기고, “추천문장 3개” 같은 건 출력 안 함
        print(json.dumps(axes, ensure_ascii=False))
        print(f"   (ok={res.get('ok')}  time={dt:.2f}s  attempt={res.get('debug', {}).get('attempt')})")

        append_log(LOG_PATH, user_text, axes, bool(res.get("ok", False)))

        wave = load_wave_summary(LOG_PATH, window=10, min_window=3)
        reply = policy_v2_friend(user_text=user_text, axes=axes)

        print("\n--- wave summary ---")
        print(json.dumps(wave, ensure_ascii=False))
        print("\n--- noiE reply (2 lines) ---")
        print(reply)


if __name__ == "__main__":
    main()

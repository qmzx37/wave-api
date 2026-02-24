# response_builder.py
from __future__ import annotations

import time
import random
from typing import Dict, List, Union, Tuple, Any, Optional

from action_space import action_pair

# ============================================================
# ✅ 타입
# ============================================================
WaveSummary = Dict[str, Union[float, str, dict]]

# ============================================================
# ✅ 스타일(모드) 상수
# ============================================================
STYLE_FRIEND = "friend"   # 친구/동료 모드(룰 기반 2줄)
STYLE_TEACH  = "teach"    # 설명/가르침 모드(나중에 확장)
STYLE_FORMAL = "formal"   # 격식 모드(나중에 확장)

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]

# (goal, style)
ActionPair = Tuple[str, str]


# ============================================================
# ✅ 값 보정 유틸
# ============================================================
def clamp01(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def normalize_axes(axes: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in KEYS:
        out[k] = clamp01(axes.get(k, 0.0))
    return out


# ============================================================
# ✅ FRIEND 정책: 2줄 고정 출력
# ============================================================
def friend_two_line(
    event: str,
    dominant: str,
    axes: dict,
    goal: str,
    moving: str = "",
    pace: str = "mid",
    style: str = "NORMAL",
) -> str:
    """
    ✅ 친구/동료 모드: 2줄 고정
    - 1줄: 친구식 미러링
    - 2줄: 질문 1개 또는 행동 1개(짧게)
    """
    axes = normalize_axes(axes)

    goal = (goal or "CLARIFY").upper()
    style = (style or "NORMAL").upper()

    # pace가 high면 더 압축된 말투
    short_by_pace = (str(pace) == "high")

    # moving: 너무 설명처럼 안 보이게 아주 짧게만
    mv = ""
    if moving == "커지는 중":
        mv = " 더 커지는 중"
    elif moving == "가라앉는 중":
        mv = " 좀 내려가는 중"

    # ✅ (기본) 설명톤 제거: 친구 말투
    if dominant == "불안":
        line1 = f"음… 불안 쪽이네.{mv}"
    elif dominant == "분노":
        line1 = f"야 그거 짜증나지.{mv}"
    elif dominant == "우울":
        line1 = f"기운 빠졌겠다…{mv}"
    elif dominant == "기쁨":
        line1 = f"오 좋은데?{mv}"
    else:
        line1 = f"오케이, 지금 좀 걸리네.{mv}"

    # event는 "원인 분석"처럼 쓰지 말고, 짧은 맥락 힌트로만
    if event and event not in ["누적 피로", "상황 정리 필요"]:
        e = event.strip()
        if len(e) > 16:
            e = e[:16] + "…"
        line1 = f"{line1} ({e})"

    # -------------------------
    # 2번째 줄: 다음 한 걸음
    # -------------------------
    if goal == "CLARIFY":
        base2 = "원하는 거 뭐야? 딱 하나만 골라봐."
        short2 = "뭐가 제일 거슬려? (하나만)"
        direct2 = "원하는 거 뭐야. 딱 하나만."
    elif goal == "STABILIZE":
        base2 = "숨 크게 2번. 그리고 뭐가 제일 거슬리는지 하나만."
        short2 = "숨 2번. 뭐가 제일 거슬려?"
        direct2 = "지금 멈추고 숨. 그다음 한 가지만 말해."
    elif goal == "MOTION":
        base2 = "2분만 하자. 물 한 컵 마시고, 할 일 1개만 적자."
        short2 = "2분만: 물 + 할 일 1개."
        direct2 = "지금 당장 하나만 해. 물 마시고, 할 일 1개 적어."
    elif goal == "AMPLIFY":
        base2 = "좋아. 지금 땡기는 거 1개만 골라봐."
        short2 = "지금 바로 할 거 1개만."
        direct2 = "바로 하나 해. 뭐부터 할래?"
    else:
        base2 = "지금 제일 큰 거 하나만 뽑자. 거기부터 가자."
        short2 = "큰 문제 1개만 뽑자."
        direct2 = "지금 하나만 찍어."

    # style/pacing 적용
    if style == "SHORT" or short_by_pace:
        line2 = short2
    elif style == "DIRECT":
        line2 = direct2
    else:
        line2 = base2

    return line1 + "\n" + line2


# ============================================================
# ✅ 내부 판단 로직
# ============================================================
def choose_mode(axes: dict) -> str:
    axes = normalize_axes(axes)
    T, R, A = axes["T"], axes["R"], axes["A"]

    if T >= 0.75 or (A >= 0.7 and R <= 0.1):
        return "SAFE"
    if R >= 0.6:
        return "SLOW"
    if T >= 0.45:
        return "FAST"
    return "BALANCED"


def choose_goal(axes: dict, event: str) -> str:
    axes = normalize_axes(axes)
    A, D, J, T, R = axes["A"], axes["D"], axes["J"], axes["T"], axes["R"]

    if A >= 0.55:
        return "CLARIFY"
    if D >= 0.55:
        return "MOTION"
    if T >= 0.55 and R <= 0.2:
        return "STABILIZE"
    if J >= 0.55:
        return "AMPLIFY"
    return "CLARIFY"


def extract_event(user_text: str) -> str:
    t = (user_text or "").strip()
    if not t:
        return "상황 정리 필요"

    for cue in ["때문에", "해서", "안 해서", "당해서", "했는데", "인데", "라서"]:
        if cue in t:
            right = t.split(cue, 1)[1].strip()
            return (right[:20] + "...") if len(right) > 20 else right

    if "돈" in t or "비용" in t:
        return "예상치 못한 비용"
    if "불안" in t or "걱정" in t:
        return "불확실한 상황"
    if "집중" in t or "딴 생각" in t:
        return "집중 방해"
    if "억울" in t or "부당" in t:
        return "부당함"

    return "누적 피로"


# ============================================================
# ✅ FRIEND 정책(파동 반영)
# ============================================================
def policy_v2_friend(
    user_text: str,
    axes: dict,
    wave_summary: WaveSummary | None = None,
) -> str:
    wave_summary = wave_summary or {}
    axes = normalize_axes(axes)

    event = extract_event(user_text)
    mode = choose_mode(axes)
    goal = choose_goal(axes, event)

    dominant_map = {"F": "불안", "A": "분노", "D": "우울", "J": "기쁨"}
    dominant_axis = max(["F", "A", "D", "J"], key=lambda k: axes.get(k, 0.0))
    dominant = dominant_map.get(dominant_axis, "상태")

    pace = str(wave_summary.get("pace", "mid"))

    delta = wave_summary.get("delta", {})
    if not isinstance(delta, dict):
        delta = {}

    loop = wave_summary.get("loop", {})
    if not isinstance(loop, dict):
        loop = {}

    loop_kind = str(loop.get("kind") or loop.get("type") or "none")

    moving = ""
    try:
        dv = float(delta.get(dominant_axis, 0.0))
        if dv > 0.05:
            moving = "커지는 중"
        elif dv < -0.05:
            moving = "가라앉는 중"
    except Exception:
        moving = ""

    if loop_kind == "anger_loop":
        goal = "CLARIFY"
    elif loop_kind == "sad_loop":
        goal = "MOTION"

    style = "NORMAL"
    if pace == "high":
        style = "SHORT"

    out = friend_two_line(
        event=event,
        dominant=dominant,
        axes=axes,
        goal=goal,
        moving=moving,
        pace=pace,
        style=style,
    )

    if mode == "SAFE":
        head = out.splitlines()[0] if out.splitlines() else out
        out = head + "\n" + "딱 한 가지만. 지금 뭐가 제일 거슬려?"

    return out


# ============================================================
# ✅ LLM messages 빌더(확장용)
# ============================================================
def build_chat_messages(
    user_text: str,
    axes: Dict[str, float],
    wave_summary: WaveSummary,
) -> List[dict]:
    axes = normalize_axes(axes)

    dom = str(wave_summary.get("dominant", "J"))
    pace = str(wave_summary.get("pace", "mid"))
    psiT = float(wave_summary.get("psi_T", axes["T"]))
    psiR = float(wave_summary.get("psi_R", axes["R"]))

    system = (
        "너는 '노이에'다. 감정 파동 기반 대화 AI다.\n"
        "- 한국어로 2~5문장\n"
        "- 판단/설교 금지\n"
        "- 요청 없으면 해결책 남발 금지\n"
        "- JSON 출력 금지"
    )

    hint = f"(파동 힌트: dom={dom}, pace={pace}, T≈{psiT:.2f}, R≈{psiR:.2f})"
    user = f"사용자 말: {user_text}\n{hint}\n규칙대로 답변해."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ============================================================
# ✅ 최종 진입점
# ============================================================
def build_response(
    user_text: str,
    axes: Dict[str, float],
    wave_summary: WaveSummary | None = None,
    style: str = STYLE_FRIEND,
) -> Union[str, List[dict]]:
    axes = normalize_axes(axes)
    wave_summary = wave_summary or {}

    if style == STYLE_FRIEND:
        return policy_v2_friend(user_text, axes, wave_summary)

    return build_chat_messages(user_text, axes, wave_summary)


# ============================================================
# ✅ NEW: 말투 다양화(템플릿 풀 + seed 샘플링)
# ============================================================
def _dominant_label(axes: Dict[str, float]) -> Tuple[str, str]:
    """
    dominant_axis: F/A/D/J (4축 중)
    dominant_label: 불안/분노/우울/기쁨
    """
    axes = normalize_axes(axes)
    dominant_map = {"F": "불안", "A": "분노", "D": "우울", "J": "기쁨"}
    dominant_axis = max(["F", "A", "D", "J"], key=lambda k: axes.get(k, 0.0))
    return dominant_axis, dominant_map.get(dominant_axis, "상태")

def _seed_bucket(stable_sec: int) -> int:
    s = max(1, int(stable_sec))
    return int(time.time() // s)

def _pick(variants: List[str], seed: str) -> str:
    if not variants:
        return ""
    rng = random.Random(seed)
    return rng.choice(variants)

def _pick_mirror_line(
    *,
    dominant: str,
    mv: str,
    event: str,
    goal: str,
    style: str,
    user_text: str,
    stable_sec: int,
) -> str:
    """
    line1(미러링) 템플릿 풀에서 선택
    """
    goal = (goal or "CLARIFY").upper()
    style = (style or "NORMAL").upper()

    # moving 문구(기본 friend_two_line과 동일한 감각)
    mv_txt = ""
    if mv == "커지는 중":
        mv_txt = " 더 커지는 중"
    elif mv == "가라앉는 중":
        mv_txt = " 좀 내려가는 중"

    # event 짧게
    e_txt = ""
    if event and event not in ["누적 피로", "상황 정리 필요"]:
        e = event.strip()
        if len(e) > 16:
            e = e[:16] + "…"
        e_txt = f" ({e})"

    # dominant별 템플릿(필요하면 여기만 계속 늘리면 됨)
    POOL: Dict[str, Dict[str, List[str]]] = {
        "불안": {
            "BASE": [
                f"음… 불안 쪽이네.{mv_txt}",
                f"지금 불안이 좀 올라온다.{mv_txt}",
                f"오케이, 불안 신호가 있네.{mv_txt}",
                f"뭔가 마음이 불편하지?{mv_txt}",
                f"불안 파동이 살짝 쎄다.{mv_txt}",
            ]
        },
        "분노": {
            "BASE": [
                f"야 그거 짜증나지.{mv_txt}",
                f"그건 화날만해.{mv_txt}",
                f"오케이, 분노가 튄다.{mv_txt}",
                f"그 상황이면 빡칠 수밖에.{mv_txt}",
                f"지금 분노 게이지 올라갔다.{mv_txt}",
            ]
        },
        "우울": {
            "BASE": [
                f"기운 빠졌겠다…{mv_txt}",
                f"오늘은 좀 가라앉아있네…{mv_txt}",
                f"마음이 무거운 쪽이야.{mv_txt}",
                f"오케이, 다운된 느낌이다…{mv_txt}",
                f"우울 파동이 슬쩍 보인다…{mv_txt}",
            ]
        },
        "기쁨": {
            "BASE": [
                f"오 좋은데?{mv_txt}",
                f"오케이, 기분 괜찮네.{mv_txt}",
                f"좋다 ㅋㅋ{mv_txt}",
                f"지금 텐션 괜찮다.{mv_txt}",
                f"기쁨 쪽으로 흐르네.{mv_txt}",
            ]
        },
        "상태": {
            "BASE": [
                f"오케이, 지금 좀 걸리네.{mv_txt}",
                f"흐름은 읽혔어.{mv_txt}",
                f"좋아, 지금 상태 확인 완료.{mv_txt}",
            ]
        },
    }

    variants = (POOL.get(dominant, POOL["상태"])).get("BASE", POOL["상태"]["BASE"])

    bucket = _seed_bucket(stable_sec)
    seed = f"L1||{user_text}||{goal}||{style}||{dominant}||{bucket}"
    line1 = _pick(variants, seed).strip() + e_txt
    return line1

def _pick_step_line(
    *,
    goal: str,
    style: str,
    pace: str,
    user_text: str,
    stable_sec: int,
) -> str:
    """
    line2(다음 한 걸음)도 템플릿 풀에서 선택
    - goal/style/pace에 따라 후보 풀 분기
    """
    goal = (goal or "CLARIFY").upper()
    style = (style or "NORMAL").upper()
    short_by_pace = (str(pace) == "high")

    POOL2: Dict[str, Dict[str, List[str]]] = {
        "CLARIFY": {
            "NORMAL": [
                "원하는 거 뭐야? 딱 하나만 골라봐.",
                "지금 원하는 포인트가 뭐야? 하나만 말해줘.",
                "오케이. 너가 얻고 싶은 결과 1개만.",
                "딱 하나만 정하자. 뭐가 제일 중요해?",
                "한 문장으로만: 지금 뭐가 제일 필요해?",
                "우선순위 1개만 찍자. 뭐야?",
            ],
            "SHORT": [
                "뭐가 제일 거슬려? (하나만)",
                "하나만 말해.",
                "포인트 1개만.",
                "핵심 1개만.",
                "지금 필요한 거 1개만.",
            ],
            "DIRECT": [
                "원하는 거 뭐야. 딱 하나만.",
                "지금 하나만 말해.",
                "돌려 말하지 말고 하나만.",
                "결론 1개만.",
            ],
        },
        "STABILIZE": {
            "NORMAL": [
                "숨 크게 2번. 그리고 뭐가 제일 거슬리는지 하나만.",
                "호흡 2번만 하고, 지금 제일 불편한 거 1개만.",
                "잠깐 멈추고 숨. 그다음 하나만 말해줘.",
                "일단 안정부터. 숨 2번 → 한 가지만.",
                "지금은 안정이 우선. 한 가지만 정하자.",
            ],
            "SHORT": [
                "숨 2번. 뭐가 제일 거슬려?",
                "호흡. 한 가지만.",
                "멈추고 숨. 하나만.",
                "숨. 그리고 1개.",
            ],
            "DIRECT": [
                "지금 멈추고 숨. 그다음 한 가지만 말해.",
                "숨부터. 그리고 하나만.",
                "안정 먼저. 하나만 말해.",
            ],
        },
        "MOTION": {
            "NORMAL": [
                "2분만 하자. 물 한 컵 마시고, 할 일 1개만 적자.",
                "지금은 작은 행동이 좋아. 2분짜리 하나만.",
                "물 한 모금 + 할 일 하나. 이 조합으로 가자.",
                "딱 2분만 움직이자. 할 일 1개만 적어.",
                "오늘은 ‘작게’ 가는 게 이겨. 하나만 하자.",
            ],
            "SHORT": [
                "2분만: 물 + 할 일 1개.",
                "작게 1개만.",
                "지금 1개만 해.",
                "2분 하나만.",
            ],
            "DIRECT": [
                "지금 당장 하나만 해. 물 마시고, 할 일 1개 적어.",
                "말 말고 행동 1개.",
                "지금 하나만 실행해.",
            ],
        },
        "AMPLIFY": {
            "NORMAL": [
                "좋아. 지금 땡기는 거 1개만 골라봐.",
                "오케이. 바로 하나만 하자. 뭐부터?",
                "지금 흐름 좋다. 다음 스텝 1개만.",
                "한 방에 가자. 지금 할 거 1개만.",
                "좋다. 지금 바로 가능한 거 하나만.",
                "오케이. TOP5에서 1픽만 해줘.",  # 장소/추천 UX에도 잘 맞음
            ],
            "SHORT": [
                "지금 바로 할 거 1개만.",
                "하나만 고르자.",
                "1개만.",
                "바로 하나.",
            ],
            "DIRECT": [
                "바로 하나 해. 뭐부터 할래?",
                "지금 1개 선택해.",
                "고민 말고 하나.",
            ],
        },
        "DEFAULT": {
            "NORMAL": [
                "지금 제일 큰 거 하나만 뽑자. 거기부터 가자.",
                "크게 하나만 잡자. 그다음이 쉬워져.",
                "우선순위 1개부터.",
            ],
            "SHORT": [
                "큰 문제 1개만.",
                "하나만.",
            ],
            "DIRECT": [
                "지금 하나만 찍어.",
                "하나만 선택해.",
            ],
        }
    }

    bucket = _seed_bucket(stable_sec)
    # pace가 high면 SHORT 강제 느낌
    eff_style = "SHORT" if (style == "SHORT" or short_by_pace) else ("DIRECT" if style == "DIRECT" else "NORMAL")

    pool_goal = POOL2.get(goal, POOL2["DEFAULT"])
    variants = pool_goal.get(eff_style, pool_goal.get("NORMAL", []))

    seed = f"L2||{user_text}||{goal}||{eff_style}||{bucket}"
    return _pick(variants, seed) or "하나만 골라줘."


# =========================
# ✅ [EXPORT FIX] PPO/Action 연결용
# - api.py가 반드시 import할 수 있게 "마지막에" 고정 정의
# - returns: (reply_text, goal, style)
# =========================
def policy_v2_friend_with_action(
    user_text: str,
    axes: dict,
    wave_summary: WaveSummary | None = None,
    action_id: Optional[int] = None,
    stable_sec: int = 20,   # ✅ NEW: 문장 변주 고정 윈도우(초)
) -> tuple[str, str, str]:
    """
    ✅ PPO(action_id) -> (goal, style) -> 2줄 FRIEND 적용
    returns: (reply_text, goal, style)

    ✅ 말투 다양화:
    - line1/line2를 goal/style/dominant 기준 템플릿 풀에서 seed 샘플링
    - stable_sec 동안은 같은 입력이면 문장 고정
    """
    wave_summary = wave_summary or {}
    axes = normalize_axes(axes)

    event = extract_event(user_text)
    mode = choose_mode(axes)

    # dominant
    dominant_axis, dominant = _dominant_label(axes)

    pace = str(wave_summary.get("pace", "mid"))

    # loop override는 goal에만 영향
    loop = wave_summary.get("loop", {})
    if not isinstance(loop, dict):
        loop = {}
    loop_kind = str(loop.get("kind") or loop.get("type") or "none")

    # moving
    delta = wave_summary.get("delta", {})
    if not isinstance(delta, dict):
        delta = {}
    moving = ""
    try:
        dv = float(delta.get(dominant_axis, 0.0))
        if dv > 0.05:
            moving = "커지는 중"
        elif dv < -0.05:
            moving = "가라앉는 중"
    except Exception:
        moving = ""

    # 1) action_id 있으면 그걸 최우선
    if action_id is not None:
        goal, style = action_pair(int(action_id))
        goal = str(goal).upper()
        style = str(style).upper()
    else:
        # 2) 없으면 기존 규칙으로
        goal = choose_goal(axes, event)
        if pace == "high" or mode == "SAFE":
            style = "SHORT"
        else:
            if axes.get("A", 0.0) >= 0.70 and axes.get("R", 0.0) <= 0.15:
                style = "DIRECT"
            else:
                style = "NORMAL"

    # loop에 따른 goal 강제
    if loop_kind == "anger_loop":
        goal = "CLARIFY"
    elif loop_kind == "sad_loop":
        goal = "MOTION"

    # ✅ 다양화된 2줄 생성
    # - SAFE 모드면 line2는 강제 고정(안정)
    line1 = _pick_mirror_line(
        dominant=dominant,
        mv=moving,
        event=event,
        goal=goal,
        style=style,
        user_text=user_text,
        stable_sec=max(1, int(stable_sec)),
    )

    if mode == "SAFE":
        line2 = "딱 한 가지만. 지금 뭐가 제일 거슬려?"
    else:
        line2 = _pick_step_line(
            goal=goal,
            style=style,
            pace=pace,
            user_text=user_text,
            stable_sec=max(1, int(stable_sec)),
        )

    reply = line1 + "\n" + line2

    return reply, goal, style

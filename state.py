from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional

KEYS = ["F", "A", "D", "J", "C", "G", "T", "R"]

# =========================
# [PATCH] topic/loop/delta 유틸
# =========================
DOM4 = ["F", "A", "D", "J"]

def _text_from_record(rec: dict) -> str:
    return str(rec.get("text", "") or "")

def _infer_topic(text: str) -> str:
    """
    ✅ 아주 단순 '관성 topic' 추정기 (제품에선 나중에 더 정교화 가능)
    """
    t = (text or "").strip()
    if not t:
        return "general"
    # 돈/규정/환불 류
    if any(k in t for k in ["돈", "값", "비용", "환불", "규정", "결제", "수수료"]):
        return "money_rule"
    # 관계/대인
    if any(k in t for k in ["사람", "친구", "가족", "연인", "회사", "상사", "동료"]):
        return "relationship"
    # 공부/집중
    if any(k in t for k in ["공부", "집중", "딴생각", "시험", "과제"]):
        return "focus_work"
    # 불안/걱정
    if any(k in t for k in ["불안", "걱정", "긴장", "초조"]):
        return "anxiety"
    # 분노/억울
    if any(k in t for k in ["화", "짜증", "억울", "부당", "열받"]):
        return "anger_unfair"
    return "general"

def _topic_from_records(records: list[dict]) -> str:
    """
    ✅ 최근 로그에서 topic 다수결로 하나 뽑기(관성)
    """
    if not records:
        return "general"
    counts: dict[str, int] = {}
    for r in records[-10:]:
        topic = _infer_topic(_text_from_record(r))
        counts[topic] = counts.get(topic, 0) + 1
    # 최빈값
    return max(counts.keys(), key=lambda k: counts[k])

def _loop_from_axes_list(axes_list: list[dict]) -> dict:
    """
    ✅ 루프 감지(간단 룰)
    - 최근 5턴에서 A>=0.60 이 3회 이상 -> anger_loop
    - 최근 5턴에서 D>=0.60 이 3회 이상 -> sad_loop
    """
    tail = axes_list[-5:] if len(axes_list) >= 1 else []
    if not tail:
        return {"kind": "none", "score": 0}

    a_cnt = sum(1 for ax in tail if float(ax.get("A", 0.0)) >= 0.60)
    d_cnt = sum(1 for ax in tail if float(ax.get("D", 0.0)) >= 0.60)

    if a_cnt >= 3:
        return {"kind": "anger_loop", "score": a_cnt}
    if d_cnt >= 3:
        return {"kind": "sad_loop", "score": d_cnt}
    return {"kind": "none", "score": 0}

def _delta_from_axes_list(axes_list: list[dict]) -> dict:
    """
    ✅ 축별 변화량 delta 계산
    - latest - prev_avg(최근 2~3개 평균)
    """
    n = len(axes_list)
    if n < 2:
        return {k: 0.0 for k in KEYS}

    latest = axes_list[-1]
    # prev window: 2개(가능하면 3개까지)
    prev_window = axes_list[max(0, n - 4): n - 1]  # 마지막 제외, 최대 3개
    m = len(prev_window) if prev_window else 1

    prev_avg: dict[str, float] = {k: 0.0 for k in KEYS}
    for ax in prev_window:
        for k in KEYS:
            prev_avg[k] += float(ax.get(k, 0.0))
    for k in KEYS:
        prev_avg[k] /= float(m)

    delta = {}
    for k in KEYS:
        delta[k] = float(latest.get(k, 0.0)) - float(prev_avg.get(k, 0.0))
    return delta

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def read_last_jsonl(path: str, n: int = 10) -> List[dict]:
    """
    ✅ emotion_log.jsonl에서 마지막 n줄만 읽어서 리스트로 반환.
    - 파일이 없거나 비어있으면 []
    """
    if not os.path.exists(path):
        return []

    # JSONL이 커질 수 있으니 "통째로 읽기" 대신,
    # 간단히 뒤에서 n줄만 가져오는 방식(메모리 안정).
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if not lines:
        return []

    tail = lines[-n:]
    out: List[dict] = []
    for s in tail:
        try:
            out.append(json.loads(s))
        except Exception:
            # 깨진 줄이 있으면 무시
            pass
    return out


def _axes_from_record(rec: dict) -> Optional[Dict[str, float]]:
    """
    JSONL 한 줄 레코드에서 labels(8축) 뽑기.
    기대 형태:
    {
      "timestamp": "...",
      "text": "...",
      "labels": {"F":..,"A":..,...},
      "meta": {...}
    }
    """
    labels = rec.get("labels")
    if not isinstance(labels, dict):
        return None

    axes: Dict[str, float] = {}
    for k in KEYS:
        axes[k] = _safe_float(labels.get(k, 0.0), 0.0)
    return axes


def l1_delta(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    ✅ 변화량(간단 버전): |a-b|의 L1 합
    - "파동이 얼마나 흔들렸는지"에 대한 스칼라 지표로 쓰기 좋음
    """
    return sum(abs(_safe_float(a.get(k, 0.0)) - _safe_float(b.get(k, 0.0))) for k in KEYS)


def summarize_wave_from_logs(records: List[dict]) -> Dict[str, Any]:
    """
    ✅ 최근 기록(3~10개)을 기반으로 파동 요약을 만든다.
    - dominant/trend/drift/pace + delta/topic/loop 포함
    - 외부 미정의 함수(_axis_name, _trend_drift_pace) 제거해서 오류 방지
    """
    axes_list: List[Dict[str, float]] = []
    for r in records:
        ax = _axes_from_record(r)
        if ax:
            axes_list.append(ax)

    n = len(axes_list)
    if n == 0:
        return {
            "n": 0,
            "dominant": "R",
            "dominant_name": "안정",
            "trend": "flat",
            "drift": 0.0,
            "pace": "low",
            "delta": {k: 0.0 for k in KEYS},
            "loop": {"kind": "none", "score": 0},
            "topic": "general",
        }

    latest = axes_list[-1]

    # ✅ dominant(상태4: F/A/D/J) 기준
    dom = max(DOM4, key=lambda k: float(latest.get(k, 0.0)))
    dom_name = {"F": "불안", "A": "분노", "D": "우울", "J": "기쁨"}[dom]

    # ✅ drift(최근 변화량 평균)
    if n == 1:
        drift = 0.0
    else:
        deltas = [l1_delta(axes_list[i - 1], axes_list[i]) for i in range(1, n)]
        drift = sum(deltas) / max(1, len(deltas))

    # ✅ pace(대충 3구간)
    if drift < 0.20:
        pace = "low"
    elif drift < 0.55:
        pace = "mid"
    else:
        pace = "high"

    # ✅ trend: dominant 축이 최근에 증가/감소 중인지
    trend = "flat"
    if n >= 3:
        x1 = float(axes_list[-3].get(dom, 0.0))
        x2 = float(axes_list[-2].get(dom, 0.0))
        x3 = float(axes_list[-1].get(dom, 0.0))
        if x3 > x2 > x1:
            trend = "up"
        elif x3 < x2 < x1:
            trend = "down"

    # ✅ [NEW] delta/topic/loop
    delta = _delta_from_axes_list(axes_list)
    loop = _loop_from_axes_list(axes_list)
    topic = _topic_from_records(records)

    return {
        "n": n,
        "dominant": dom,
        "dominant_name": dom_name,
        "trend": trend,
        "drift": float(drift),
        "pace": pace,
        "delta": delta,
        "loop": loop,
        "topic": topic,
    }


def load_wave_summary(
    log_path: str,
    window: int = 10,
    min_window: int = 3,
) -> Dict[str, Any]:
    """
    ✅ log에서 최근 window개를 읽고 파동 요약.
    - 최소 min_window개 정도 확보하려고 노력
    """
    recs = read_last_jsonl(log_path, n=window)
    # 혹시 너무 적으면 그대로 사용
    if len(recs) < min_window:
        return summarize_wave_from_logs(recs)
    return summarize_wave_from_logs(recs)

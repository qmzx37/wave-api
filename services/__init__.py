# action_space.py
from __future__ import annotations

from typing import Dict, Tuple

# 4 goals × 3 styles = 12 actions
GOALS = ["CLARIFY", "STABILIZE", "MOTION", "AMPLIFY"]
STYLES = ["NORMAL", "SHORT", "DIRECT"]

ActionPair = Tuple[str, str]

NUM_ACTIONS: int = len(GOALS) * len(STYLES)

PAIR_BY_ACTION_ID: Dict[int, ActionPair] = {}
ACTION_ID_BY_PAIR: Dict[ActionPair, int] = {}

for gi, g in enumerate(GOALS):
    for si, s in enumerate(STYLES):
        aid = gi * len(STYLES) + si
        pair = (g, s)
        PAIR_BY_ACTION_ID[aid] = pair
        ACTION_ID_BY_PAIR[pair] = aid


def _norm_goal(x: str) -> str:
    x = (x or "").strip().upper()
    return x if x in GOALS else "CLARIFY"


def _norm_style(x: str) -> str:
    x = (x or "").strip().upper()
    return x if x in STYLES else "NORMAL"


def action_id(goal: str, style: str) -> int:
    """(goal, style) -> action_id (0..11)"""
    g = _norm_goal(goal)
    s = _norm_style(style)
    return int(ACTION_ID_BY_PAIR.get((g, s), 0))


def action_pair(action: int) -> ActionPair:
    """action_id -> (goal, style)"""
    try:
        a = int(action)
    except Exception:
        a = 0
    return PAIR_BY_ACTION_ID.get(a, ("CLARIFY", "NORMAL"))

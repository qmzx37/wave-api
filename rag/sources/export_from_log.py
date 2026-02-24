from __future__ import annotations

import os, json, hashlib, datetime
from typing import Any, Dict, List

LOG_DEFAULT = r"C:\llm\train\emotion_log.jsonl"


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def export_from_log(
    log_path: str = LOG_DEFAULT,
    out_dir: str = os.path.join("rag", "data", "log"),
    max_lines: int = 2000,
    group: str = "action",   # "action" or "day"
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(log_path):
        return {"ok": False, "error": f"log not found: {log_path}"}

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lines = lines[-max_lines:]

    rows: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        ts = _safe_str(r.get("timestamp", ""))
        text = _safe_str(r.get("text", ""))
        reply = _safe_str(r.get("reply", ""))
        meta = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}

        # action id / goal
        action = None
        if isinstance(meta.get("action", None), dict):
            action = meta["action"]
        # 어떤 로그는 meta.wave_state.action에 있을 수도 있음
        if action is None and isinstance(meta.get("wave_state", None), dict):
            ws = meta["wave_state"]
            if isinstance(ws.get("action", None), dict):
                action = ws["action"]

        action_id = None
        goal = ""
        if isinstance(action, dict):
            action_id = action.get("id", None)
            goal = _safe_str(action.get("goal", ""))

        if group == "day":
            key = ts[:10] if len(ts) >= 10 else "unknown"
        else:
            key = f"action_{action_id}_{goal}".strip("_")

        buckets.setdefault(key, []).append({
            "timestamp": ts,
            "text": text,
            "reply": reply,
            "action_id": action_id,
            "goal": goal,
        })

    saved = 0
    for key, items in buckets.items():
        # 문서 본문 구성: 질문/응답을 “기록”으로 묶기
        body_lines = []
        for it in items:
            t = it["text"].strip()
            rp = it["reply"].strip()
            if not t:
                continue
            body_lines.append(f"- U: {t}")
            if rp:
                body_lines.append(f"  A: {rp}")
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        doc = {
            "id": _sha1(key + "::" + body[:2000]),
            "url": f"local://emotion_log/{key}",
            "source": "emotion_log",
            "title": f"emotion_log bucket: {key}",
            "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "text": body,
            "meta": {
                "bucket": key,
                "count": len(items),
                "group": group,
            },
        }

        out_path = os.path.join(out_dir, f"{doc['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        saved += 1

    return {"ok": True, "saved": saved, "out_dir": out_dir, "groups": len(buckets), "max_lines": max_lines}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", default=LOG_DEFAULT)
    p.add_argument("--out_dir", default=os.path.join("rag", "data", "log"))
    p.add_argument("--max_lines", type=int, default=2000)
    p.add_argument("--group", default="action", choices=["action", "day"])
    args = p.parse_args()

    print(export_from_log(args.log_path, args.out_dir, args.max_lines, args.group))
from __future__ import annotations

import os, json, hashlib, datetime
from typing import Any, Dict, List

LOG_DEFAULT = r"C:\llm\train\emotion_log.jsonl"


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def export_from_log(
    log_path: str = LOG_DEFAULT,
    out_dir: str = os.path.join("rag", "data", "log"),
    max_lines: int = 2000,
    group: str = "action",   # "action" or "day"
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(log_path):
        return {"ok": False, "error": f"log not found: {log_path}"}

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lines = lines[-max_lines:]

    rows: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue

    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        ts = _safe_str(r.get("timestamp", ""))
        text = _safe_str(r.get("text", ""))
        reply = _safe_str(r.get("reply", ""))
        meta = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}

        # action id / goal
        action = None
        if isinstance(meta.get("action", None), dict):
            action = meta["action"]
        # 어떤 로그는 meta.wave_state.action에 있을 수도 있음
        if action is None and isinstance(meta.get("wave_state", None), dict):
            ws = meta["wave_state"]
            if isinstance(ws.get("action", None), dict):
                action = ws["action"]

        action_id = None
        goal = ""
        if isinstance(action, dict):
            action_id = action.get("id", None)
            goal = _safe_str(action.get("goal", ""))

        if group == "day":
            key = ts[:10] if len(ts) >= 10 else "unknown"
        else:
            key = f"action_{action_id}_{goal}".strip("_")

        buckets.setdefault(key, []).append({
            "timestamp": ts,
            "text": text,
            "reply": reply,
            "action_id": action_id,
            "goal": goal,
        })

    saved = 0
    for key, items in buckets.items():
        # 문서 본문 구성: 질문/응답을 “기록”으로 묶기
        body_lines = []
        for it in items:
            t = it["text"].strip()
            rp = it["reply"].strip()
            if not t:
                continue
            body_lines.append(f"- U: {t}")
            if rp:
                body_lines.append(f"  A: {rp}")
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        doc = {
            "id": _sha1(key + "::" + body[:2000]),
            "url": f"local://emotion_log/{key}",
            "source": "emotion_log",
            "title": f"emotion_log bucket: {key}",
            "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "text": body,
            "meta": {
                "bucket": key,
                "count": len(items),
                "group": group,
            },
        }

        out_path = os.path.join(out_dir, f"{doc['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        saved += 1

    return {"ok": True, "saved": saved, "out_dir": out_dir, "groups": len(buckets), "max_lines": max_lines}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log_path", default=LOG_DEFAULT)
    p.add_argument("--out_dir", default=os.path.join("rag", "data", "log"))
    p.add_argument("--max_lines", type=int, default=2000)
    p.add_argument("--group", default="action", choices=["action", "day"])
    args = p.parse_args()

    print(export_from_log(args.log_path, args.out_dir, args.max_lines, args.group))

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParsedAction:
    action: str  # "tool_call" | "final"
    payload: Dict[str, Any]


def extract_outermost_json_object(text: str) -> Optional[str]:
    """
    Extract the first *balanced* JSON object substring from text.
    Robust against leading/trailing chatter.

    Notes:
    - Handles nested braces.
    - Attempts to ignore braces inside JSON strings.
    """
    if not text:
        return None

    s = text
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def _strip_code_fences(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) <= 1:
        return s
    if lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _normalize_model_json_text(text: str) -> str:
    """
    Remove common reasoning wrappers before JSON parsing.
    Handles MedGemma thought markers and generic <think> wrappers.
    """
    s = str(text or "")
    s = re.sub(r"(?is)<think>.*?</think>", "", s)
    if "<unused95>" in s:
        # MedGemma often puts final answer after this delimiter.
        s = s.rsplit("<unused95>", 1)[-1]
    s = s.replace("<unused94>thought", "")
    s = s.replace("<unused95>", "")
    s = _strip_code_fences(s)
    return s.strip()


def _json_object_fragments(text: str) -> List[str]:
    """
    Enumerate balanced JSON object fragments from a text blob.
    """
    s = str(text or "")
    out: List[str] = []
    seen = set()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        frag = extract_outermost_json_object(s[i:])
        if not frag:
            continue
        key = frag.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _json_dict_candidates(text: str) -> List[Dict[str, Any]]:
    """
    Parse all plausible dict candidates from text, in priority order.
    """
    raw = str(text or "")
    normalized = _normalize_model_json_text(raw)
    pools: List[str] = []
    if normalized:
        pools.append(normalized)
    if raw and raw != normalized:
        pools.append(raw)

    out: List[Dict[str, Any]] = []
    seen = set()

    def _add_obj(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        try:
            key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = str(obj)
        if key in seen:
            return
        seen.add(key)
        out.append(obj)

    for p in pools:
        try:
            _add_obj(json.loads(p))
        except Exception:
            pass
        for frag in _json_object_fragments(p):
            try:
                _add_obj(json.loads(frag))
            except Exception:
                continue
    return out


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing:
    - direct json.loads(text)
    - else extract balanced {...} and parse
    """
    candidates = _json_dict_candidates(text)
    return candidates[0] if candidates else None


def validate_tool_action(obj: Dict[str, Any]) -> ParsedAction:
    """
    Validate model output into a minimal action contract.

    Supported actions:
    - tool_call: {"action":"tool_call","tool_name":str,"arguments":dict, "stage"?:str}
    - tool_calls: {"action":"tool_calls","calls":[{tool_call}, ...]}
    - final: {"action":"final","final_report":<any>}
    """
    action = obj.get("action")
    if action in ("tools/call", "call_tool"):
        action = "tool_call"
    if action not in ("tool_call", "tool_calls", "final"):
        raise ValueError("Missing/invalid 'action'. Expected 'tool_call', 'tool_calls', or 'final'.")

    if action == "tool_call":
        tool_name = obj.get("tool_name")
        arguments = obj.get("arguments")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_call requires non-empty 'tool_name' (string).")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call requires 'arguments' (object/dict).")
        stage = obj.get("stage", "misc")
        if stage is not None and not isinstance(stage, str):
            raise ValueError("'stage' must be a string if provided.")
        return ParsedAction(action="tool_call", payload={"tool_name": tool_name, "arguments": arguments, "stage": stage or "misc"})

    if action == "tool_calls":
        calls = obj.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls requires non-empty 'calls' (array).")
        norm_calls = []
        for i, c in enumerate(calls):
            if not isinstance(c, dict):
                raise ValueError(f"calls[{i}] must be an object/dict.")
            tool_name = c.get("tool_name")
            arguments = c.get("arguments")
            stage = c.get("stage", "misc")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"calls[{i}].tool_name must be a non-empty string.")
            if not isinstance(arguments, dict):
                raise ValueError(f"calls[{i}].arguments must be an object/dict.")
            if stage is not None and not isinstance(stage, str):
                raise ValueError(f"calls[{i}].stage must be a string if provided.")
            norm_calls.append({"tool_name": tool_name, "arguments": arguments, "stage": stage or "misc"})
        return ParsedAction(action="tool_calls", payload={"calls": norm_calls})

    # final
    return ParsedAction(action="final", payload={"final_report": obj.get("final_report", obj.get("report", obj))})


def parse_model_output(text: str) -> Tuple[Optional[ParsedAction], Optional[str]]:
    """
    Parse + validate. Returns (ParsedAction|None, error_message|None).
    """
    candidates = _json_dict_candidates(text)
    if not candidates:
        return None, "Could not parse any JSON object from model output."
    first_err: Optional[str] = None
    for obj in candidates:
        try:
            act = validate_tool_action(obj)
            return act, None
        except Exception as e:
            if first_err is None:
                first_err = str(e)
            continue
    return None, f"JSON parsed but failed validation: {first_err or 'unknown validation error'}"

"""Shared helpers: JSONL I/O, answer extraction, normalisation."""

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


# ----------------------------------------------------------------------------
# JSONL
# ----------------------------------------------------------------------------
def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows: Iterable[dict], mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    write_jsonl(path, rows, mode="a")


# ----------------------------------------------------------------------------
# Answer extraction
# ----------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BRACES_RE = re.compile(r"\{[^{}]*\}")


def strip_thinking(text: str) -> str:
    """Remove a leading <think>...</think> block (reasoning models)."""
    if text is None:
        return ""
    if "<think>" in text:
        text = _THINK_RE.sub("", text)
        # unterminated think block -> nothing usable after it
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>")[0]
    return text


def _try_parse(s: str) -> Optional[dict]:
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return None


def extract_json_answer(text: str, key: str, last: bool = True):
    """Port of MMTU's ``BaseEvaluator.extract_json_answer``: find the
    ``{...}`` fragments in ``text`` and return ``obj[key]`` of the first (or
    last, when ``last=True``) fragment that parses to a dict containing ``key``.

    MMTU takes the first match. For chain-of-thought outputs (NIAT cell-lookup)
    the final answer is the last JSON block, so ``last=True`` is the default
    here; for the MMTU tasks the evaluator passes ``last=False`` to stay faithful.
    Returns the sentinel string ``"JSONParsingError"`` when nothing parses.
    """
    text = strip_thinking(text or "")
    matches = _BRACES_RE.findall(text)
    if last:
        matches = matches[::-1]
    for m in matches:
        obj = _try_parse(m)
        if obj is not None and key in obj:
            return obj[key]
    # also handle doubled braces "{{ ... }}" which some models copy from the prompt
    if "{{" in text:
        return extract_json_answer(text.replace("{{", "{").replace("}}", "}"), key, last)
    return "JSONParsingError"


# ----------------------------------------------------------------------------
# Normalisation for exact match
# ----------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def norm_strict(v: Any) -> str:
    """Exact match after stripping surrounding whitespace."""
    return to_text(v).strip()


def norm_loose(v: Any) -> str:
    """Case-insensitive, whitespace-collapsed, quote-stripped comparison."""
    s = to_text(v).strip().lower()
    s = _WS_RE.sub(" ", s)
    s = s.strip("\"'`")
    return s


def numeric_equal(a: Any, b: Any) -> bool:
    try:
        return float(to_text(a).replace(",", "")) == float(to_text(b).replace(",", ""))
    except Exception:
        return False

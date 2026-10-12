"""NIAT scoring: exact match, as in the paper.

* cell_locating / cropped: answer JSON ``{"cell_value": v}``.
* cell_lookup: chain of thought followed by ``{"answer": v}`` (the last JSON block counts).
"""

from .utils import norm_loose, norm_strict, numeric_equal

ANSWER_KEY = {"cell_locating": "cell_value", "cropped": "cell_value", "cell_lookup": "answer"}


def score(gt, pred) -> dict:
    """``em``: exact match after stripping surrounding whitespace (the reported metric);
    ``em_loose``: case-insensitive, whitespace-collapsed, quotes stripped, or numerically equal."""
    if pred == "JSONParsingError":
        return {"em": 0, "em_loose": 0}
    em = int(norm_strict(gt) == norm_strict(pred))
    loose = int(em or norm_loose(gt) == norm_loose(pred) or numeric_equal(gt, pred))
    return {"em": em, "em_loose": loose}

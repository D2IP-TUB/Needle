"""Named configurations of ``solve.py``.

A preset is a set of ``CCS_<KNOB>`` values. ``CCS_PRESET=<name>`` selects one; any ``CCS_<KNOB>``
set in the environment overrides the preset. Every knob is documented next to its default in
``solve.py``.

The presets fall into three groups:

* the CCS system in its tag shapes (``ccs`` is the default when no preset is given);
* the cumulative arms of the ablation and transfer runs (``baseline`` -> ``tags_only`` ->
  ``tags_protocol`` -> ``full``), which use the per-task tag shapes;
* the serialization baselines (``markdown``, ``html``, ``csv``, ``json``, ...), each with and without the
  format-independent parts of the answer protocol (suffix ``_protocol``), and SpreadsheetLLM.
"""

# everything off: the published prompt, greedy decoding, nothing added
_OFF = dict(TAG_SHAPE="per_task", TAG_ANSWER_SETS="", NEEDLE_TAG="",
            CELL_TAGS=0, NIAT_CELL_TAGS=0, CODE_COLUMNS=0, LOCATE_CODES=0, COLUMNS_ONE_PER_LINE=0,
            VALUE_FIRST_SETS="", JSON_ONLY_HINT=0, QUERY_FIRST=0, QUERY_LAST_MMTU=0, COPY_EXACT_HINT=0,
            CONSTRAIN_CELL_LOCATING=0, CONSTRAIN_LOCATE_TAG=0, CONSTRAIN_CROPPED=0, RETRY_NO_JSON="")

# the parts of the answer protocol that do not depend on how the table is written: the query repeated on the
# far side of the table, the "JSON only" / "copy the cell exactly" hints, one retry on cell-lookup when no JSON
# object was produced, and the cell-locating answer limited to the table's own cell strings
_PROTOCOL = dict(JSON_ONLY_HINT=1, QUERY_FIRST=1, QUERY_LAST_MMTU=1, COPY_EXACT_HINT=1,
                 RETRY_NO_JSON="niat_cell_lookup", CONSTRAIN_CELL_LOCATING=1)

PRESETS = {
    # ---- CCS, one tag shape for every task (the method of the paper) ----
    "ccs": {},                                              # [r<row>|c<n>=value]
    "ccs_colword": dict(TAG_SHAPE="colword"),               # [row <row>, column c<n> = value]
    "ccs_valuefirst_needle": dict(NEEDLE_TAG="valuefirst"),  # [r|c=value] everywhere, [value @ row <row>, c<n>] on the needle task
    # ---- CCS with the per-task tag shapes, as cumulative arms (ablation and transfer tables) ----
    "baseline": dict(_OFF, SERIALIZATION="baseline"),
    "tags_only": dict(_OFF, SERIALIZATION="ccs", CELL_TAGS=1, NIAT_CELL_TAGS=1),
    "tags_protocol": dict(_OFF, SERIALIZATION="ccs", CELL_TAGS=1, NIAT_CELL_TAGS=1, CODE_COLUMNS=1,
                          LOCATE_CODES=1, COLUMNS_ONE_PER_LINE=1, VALUE_FIRST_SETS="mmtu_niht",
                          JSON_ONLY_HINT=1, QUERY_FIRST=1, QUERY_LAST_MMTU=1, COPY_EXACT_HINT=1,
                          RETRY_NO_JSON="niat_cell_lookup"),
}
PRESETS["full"] = dict(PRESETS["tags_protocol"], CONSTRAIN_CELL_LOCATING=1, CONSTRAIN_LOCATE_TAG=1, CONSTRAIN_CROPPED=1)
PRESETS["ccs_per_task"] = PRESETS["full"]

# ---- serialization baselines: the published prompt with only the table block swapped ----
FORMATS = ["markdown", "html", "csv", "json", "jsonl", "html_df", "html_own", "spreadsheetllm", "spreadsheetllm_inv"]
for _f in FORMATS:
    PRESETS[_f] = dict(_OFF, SERIALIZATION=_f)
    PRESETS[_f + "_protocol"] = dict(_OFF, SERIALIZATION=_f, **_PROTOCOL)
# the tags with the same format-independent parts (the "tags + protocol" row of the main table)
PRESETS["tags_only_protocol"] = dict(PRESETS["tags_only"], **_PROTOCOL)

ARMS = ["baseline", "tags_only", "tags_protocol", "full"]


def preset(name):
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'. Known: {sorted(PRESETS)}")
    return PRESETS[name]


def env_for(name, **extra):
    """``{"CCS_<KNOB>": value}`` for a preset, to be merged into a subprocess environment."""
    out = {"CCS_PRESET": name}
    out.update({f"CCS_{k}": str(v) for k, v in extra.items()})
    return out

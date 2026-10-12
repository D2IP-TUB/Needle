"""Checks that need no GPU: the CCS serialization of a small table, the grammars, and the scoring.

    python -m pytest tests/        (or: python tests/test_transform.py)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CCS_PRESET", "ccs")

import benchmark  # noqa: E402
import solve  # noqa: E402

TABLE = ("| city | country | population |\n"
         "| --- | --- | --- |\n"
         "| Oslo | Norway | 0.71 |\n"
         "| Kyoto | Japan | 1.46 |")
NIAT_PROMPT = ("You are an expert in analyzing and extracting information from tabular data. "
               "Output the result as a JSON object in the following format:\n"
               '{"row_ID": "m", "column_ID": "n", "cell_value": "<Corresponding Cell Value>"}\n\n'
               "Given table:\n" + TABLE + "\n\n\n"
               "Now, retrieve the cell value at the position where the row_ID is 2 and the column_ID is 1.\n\nThe answer is:\n")


def test_ccs_table_matches_the_paper_figure():
    """The tagged table of the paper's figure: column list, header as row 1, [r<row>|c<n>=value]."""
    out = solve.transform(NIAT_PROMPT, "niat_cell_locating")
    assert "columns (name -> code):\n  city -> c1\n  country -> c2\n  population -> c3\nrows:\n" in out
    assert "row 1 (header): [r1|c1=city] [r1|c2=country] [r1|c3=population]" in out
    assert "row 2: [r2|c1=Oslo] [r2|c2=Norway] [r2|c3=0.71]" in out
    assert "row 3: [r3|c1=Kyoto] [r3|c2=Japan] [r3|c3=1.46]" in out
    assert "Target cell (repeated after the table): row_ID 2, column_ID 1." in out
    assert '"tag": "[r<m>|c<n>=<cell>]"' in out            # the answer copies the tag before the value


def test_grammar_admits_only_real_tags():
    import re
    regexes, tag_maps = solve.grammars([NIAT_PROMPT], "niat_cell_locating")
    rx = re.compile(regexes[0])
    assert rx.fullmatch('{"row_ID": "2", "column_ID": "1", "tag": "[r2|c1=Oslo]", "cell_value": "Oslo"}')
    assert not rx.fullmatch('{"row_ID": "2", "column_ID": "1", "tag": "[r2|c1=Paris]", "cell_value": "Paris"}')
    assert tag_maps[0]["[r3|c2=Japan]"] == "Japan"
    fixed = solve.value_from_tag('{"row_ID": "2", "column_ID": "1", "tag": "[r2|c1=Oslo]", "cell_value": "Osl"}', tag_maps[0], "niat_cell_locating")
    assert json.loads(fixed)["cell_value"] == "Oslo"


def test_scoring_follows_the_benchmarks():
    rec = {"gt": {"row": 12, "column": "Population"}}
    assert benchmark.score_one("mmtu_niht", rec, '{"row": 12, "column": "population"}') == 1   # column: case-insensitive
    assert benchmark.score_one("mmtu_niht", rec, '{"row": "13", "column": "population"}') == 0
    assert benchmark.score_one("mmtu_table_locate", {"gt": "1.50"}, '{"cell": 1.5}') == 1       # numeric match
    assert benchmark.score_one("niat_cropped", {"gt": "Oslo"}, 'text {"cell_value": "Oslo"}') == 1
    assert benchmark.score_one("niat_cropped", {"gt": "Oslo"}, 'text {"cell_value": "oslo"}') == 0   # exact match
    m = benchmark.aggregate({"mmtu_niht": 1.0, "mmtu_table_locate": 0.5, "niat_cell_locating": 1.0, "niat_cropped": 1.0, "niat_cell_lookup": 0.4})
    assert abs(m["score"] - 0.775) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

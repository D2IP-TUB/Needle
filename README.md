# Coordinate-carrying serialization (CCS) for cell retrieval in large tables

Code and results for the paper *Needle Found: How Coordinate-Carrying Serialization Solves Cell Retrieval in
Large Tables*.

Large language models retrieve a needle from long text almost perfectly, yet fail to retrieve a single cell from
a large table: to reach a cell's column in a Markdown, CSV, HTML or JSON table the model has to count delimiters,
and that count loses its resolution beyond roughly the sixteenth position. CCS writes every cell as a bracketed
tag that holds its row identifier, its column identifier and its value, so that the position of a value is a
local property of the value rather than something to be counted:

```
columns (name -> code):
  city -> c1
  country -> c2
  population -> c3
rows:
row 1 (header): [r1|c1=city]  [r1|c2=country] [r1|c3=population]
row 2:          [r2|c1=Oslo]  [r2|c2=Norway]  [r2|c3=0.71]
row 3:          [r3|c1=Kyoto] [r3|c2=Japan]   [r3|c3=1.46]
```

The method is a prompt transformation plus an answer protocol (fixed JSON answer format whose fields follow
the lookup, the query repeated on the far side of the table) and constrained decoding of the copied fields
(XGrammar regular expressions built from the table). No training, no model access, every call sees every cell.

Everything runs on local open-weight models with vLLM. The main model is Qwen3-4B-Instruct-2507; eight further
models are used for the transfer table.

## Layout

```
solve.py                 CCS: serialization, answer protocol, constrained decoding; evaluation on the subset
benchmark.py             the five prompt sets, the fixed evaluation subset, the benchmarks' metrics
ccs/                     package: data loaders (MMTU, NIAT), evaluator ports, model registry, vLLM wrapper,
                         named configurations (presets.py), Table-QA helpers
scripts/                 build the prompt sets, run the subset / full-set experiments, scoring, tables
mechanistic/             the analyses of the mechanistic section (attention, probes, head roles, knock-outs)
figures/                 the figure scripts (read results/, write figures/out/)
results/                 the result files behind every table and figure of the paper (see below)
data/niat/               the 360 NIAT source tables and released samples; data/subset_ids.json pins the subset
tests/                   CPU checks of the serialization, the grammars and the scoring
```

## Setup

Python 3.11 with vLLM 0.13, PyTorch 2.9, transformers 4.57 and XGrammar 0.1.27 (``requirements.txt``). The
model weights are pulled from the Hugging Face hub (``ccs/models.py`` lists the ids); the runs of the paper used
one H200 GPU per model.

```bash
pip install -r requirements.txt
python scripts/build_data.py        # full prompt sets -> data/processed (downloads MMTU from the hub)
python benchmark.py --build         # evaluation subset -> data/benchmark (checked against data/subset_ids.json)
python tests/test_transform.py      # no GPU
```

### Prompt sets

| set | task | full | subset |
|---|---|---:|---:|
| `mmtu_niht` | MMTU needle, value -> position | 999 | 150 |
| `mmtu_table_locate` | MMTU table-locate, position -> value | 1,000 | 150 |
| `niat_cell_locating` | NIAT cell-locating, position -> value | 14,396 | 600 |
| `niat_cropped` | NIAT fixed-size tables 8x8 .. 32x32 | 48,720 | 6,496 |
| `niat_cell_lookup` | NIAT cell-lookup, question -> value | 14,054 | 600 |

MMTU prompts are used verbatim from the Hugging Face dataset. NIAT released its tables and lookup questions but
not the full query sets; they are regenerated from the released tables with the templates of the released
samples (``ccs/niat_data.py``). Metrics are the benchmarks' own (MMTU accuracy with ports of its evaluators,
NIAT exact match); the aggregate score is the mean of the MMTU accuracy (mean of its two tasks) and the NIAT
exact match (mean of its three tasks). Prompts longer than a model's context count as wrong.

## Configurations

``solve.py`` is driven by knobs ``CCS_<KNOB>`` in the environment; ``CCS_PRESET=<name>`` selects a named
configuration from ``ccs/presets.py``, single knobs override it. The configurations behind the paper's numbers:

| preset | what it is | where it appears |
|---|---|---|
| `ccs` (default) | one tag shape for every task, `[r<row>\|c<n>=value]`, with the answer protocol and constrained decoding | the method of the paper |
| `ccs_colword` | the same with the coordinates spelled out, `[row <row>, column c<n> = value]` | Table 1, CCS row, needle task (full set) |
| `ccs_valuefirst_needle` | `ccs` with the needle task written value-first, `[value @ row <row>, c<n>]` | Table 1, CCS row, the four other sets (full sets) |
| `full` (= `ccs_per_task`) | one tag shape per task direction: value-first on the needle task, `[r\|c<n>\|name=value]` on table-locate, `[r\|c<n>=value]` on the NIAT sets | Table 2 last row, Table 3 CCS column, Table 4 (tokens), the position heat maps, the mechanistic section |
| `tags_only`, `tags_protocol` | the cumulative arms before `full`: tags alone; tags plus the answer protocol | Table 2 |
| `baseline` | the published prompt, nothing changed | reference |
| `markdown_protocol`, `html_own_protocol`, `csv_protocol`, `jsonl_protocol` | the common serializations under the format-independent parts of the answer protocol | Table 1 baseline rows, Table 3 Markdown and JSON columns, heat maps |
| `markdown`, `html`, `csv`, `json`, `jsonl`, `html_df`, `html_own` | the common serializations alone | reference |
| `spreadsheetllm`, `spreadsheetllm_inv` | SpreadsheetLLM's encoding without and with its inverted index | Table 1 |

On the NIAT position -> value sets the `ccs*` presets and `full` write the same tag; on the needle task the
full-set run of the paper's Table 1 uses the `ccs_colword` shape (0.948), and the `ccs` shape gives 0.911,
`ccs_valuefirst_needle` 0.962 and the per-task value-first form 0.979 on the same 999 prompts (``results/fullset``).

## Reproducing the experiments

All GPU scripts take ``CUDA_VISIBLE_DEVICES``; the runners put one job per GPU.

```bash
# Table 2 (ablation) and Table 3 (transfer): nine models x four arms on the evaluation subset
python scripts/run_subset.py --gpus 0 1 2
# Table 1 and 3 baselines: common serializations under the answer protocol
python scripts/run_subset.py --gpus 0 1 2 --presets markdown_protocol html_own_protocol csv_protocol jsonl_protocol
python scripts/run_subset.py --gpus 0 --models qwen3-4b-2507 --presets spreadsheetllm spreadsheetllm_inv
python scripts/score_spreadsheetllm.py results/subset/outputs/qwen3-4b-2507__spreadsheetllm.jsonl   # sheet coordinates mapped

# Table 1, CCS on the full sets (79,169 prompts; about five hours per configuration on one H200)
CUDA_VISIBLE_DEVICES=0 python scripts/run_fullset.py                                   # ccs, all five sets
CCS_PRESET=ccs_colword CUDA_VISIBLE_DEVICES=1 python scripts/run_fullset.py mmtu_niht
CCS_PRESET=ccs_valuefirst_needle CUDA_VISIBLE_DEVICES=2 python scripts/run_fullset.py
python scripts/score_fullset.py results/fullset/qwen3-4b-2507__ccs results/fullset/qwen3-4b-2507__full

# Table 4 (prompt length) and Table 5 (MMTU Table-QA)
CCS_PRESET=full python scripts/count_tokens.py
CUDA_VISIBLE_DEVICES=0 python scripts/mmtu_table_qa.py --variants reason_markdown reason_json reason_tagged

# the tables of the paper from results/
python scripts/tables.py
```

A single configuration on the subset: ``CUDA_VISIBLE_DEVICES=0 python solve.py`` (about 50 minutes for the
default on one H200; ``CCS_LIMIT=4`` for a smoke test).

### Mechanistic section (Qwen3-4B-Instruct-2507, Hugging Face transformers, eager attention)

| script | what it computes | paper |
|---|---|---|
| `mechanistic/attention_niat.py` | cell-to-cell and answer-position attention on an 8x8 NIAT table, Markdown / JSON / CCS | attention figures |
| `mechanistic/attention_crops.py` | the same table at 16x16 and 32x32, with the accuracy on every query | appendix figures |
| `mechanistic/size_sweep.py a..f` | ten serializations x table sizes, 100 synthetic tables per size; `table` prints the LaTeX rows | size x serialization table |
| `mechanistic/probe.py` | ridge probes for the row and column index at layer 17, held-out tables of 16 to 128 | probe figure and numbers |
| `mechanistic/count_code.py` | the shape of the count code along the layers (a + b/j fits) | the 1/j paragraph |
| `mechanistic/head_roles.py` | attention roles of every head under the three serializations; head populations | role maps, head bars |
| `mechanistic/stages.py` | single heads on the 5x5 orders table | stage figure |
| `mechanistic/ablation.py` | head knock-outs with random controls: count code, mover attention, accuracy | causal numbers, ablation table |

Outputs go to ``results/mechanistic/<topic>/``; the figures are drawn from there:

```bash
python figures/fig_position_heatmaps.py       # MMTU needle accuracy by position, five serializations
python figures/fig_attention.py               # cells8 cells16 cells32 answer layers
python figures/fig_mech.py                    # probe roles heads stages numbers
```

## Results

The runners and analyses write ``results/`` in the layout below, which ``scripts/tables.py`` and the figure
scripts read (no GPU needed once the result files exist). The result files themselves are not part of the
repository:

* ``subset/<model>__<preset>.json``: metrics of every (model, preset) run on the evaluation subset, plus the
  SpreadsheetLLM outputs in ``subset/outputs``; ``full`` runs carry the NIAT cropped set from the rerun with
  constrained decoding on that set.
* ``fullset/<model>__<preset>/metrics_<set>.json``: full-set runs; the needle outputs of the heat-map arms.
* ``table_qa/``: MMTU Table-QA summary. ``tokens_full.json``: prompt lengths.
* ``mechanistic/``: the measurements behind the mechanistic figures (attention maps, probe read-outs, head roles
  and populations, the single-head attention of the stage figure, the knock-outs, the size sweep).

Raw generations are written next to the metrics (``outputs/``).

## Notes

* Decoding is greedy everywhere; output budgets are 1,024 / 512 tokens on the MMTU tasks and 2,048 on the NIAT
  tasks. Qwen3-4B-Instruct-2507 runs with its native 256K window; the other models at their own maximum
  (Qwen2.5 and Qwen3-8B extended to 128K with YaRN as documented on their model cards). Mistral-7B's 32K window
  holds no MMTU prompt, so its MMTU accuracy is 0 in every configuration.
* ``ccs/mmtu_eval.py`` ports MMTU's ``tniah_evaluator`` and ``tablelocate_evaluator`` (from
  https://github.com/MMTU-Benchmark/MMTU); NIAT is scored by exact match on the ``cell_value`` / ``answer`` field.
* The mechanistic scripts use the per-task tag shape on NIAT prompts (``CCS_PRESET=full``), which is the CCS
  form the paper analyses; it is identical to the default shape on those sets except for the column list.

"""Coordinate-carrying serialization (CCS) for cell retrieval in large tables.

The package holds the data loaders of the two benchmarks (MMTU needle tasks, NeedleInATable),
faithful ports of their evaluators, the model registry and the vLLM wrapper. The method itself
(serialization, answer protocol, constrained decoding) lives in ``solve.py`` at the repository root.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

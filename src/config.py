"""Portable project paths and the default reproducibility seed."""
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
RANDOM_STATE: Final[int] = 42

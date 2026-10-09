"""Verify the Stage 1 project structure without loading data or training models."""
import platform
from pathlib import Path

from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR


def required_directories() -> tuple[Path, ...]:
    """Return directories required by the initial project layout."""
    return (
        PROJECT_ROOT / "data", RAW_DATA_DIR, PROCESSED_DATA_DIR,
        PROJECT_ROOT / "src", PROJECT_ROOT / "notebooks",
        RESULTS_DIR, PROJECT_ROOT / "tests",
    )


def verify_project_directories() -> None:
    """Raise an informative error if any required directory is missing."""
    missing = [str(path) for path in required_directories() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing required project directories: " + ", ".join(missing))


def main() -> None:
    """Display environment information and verify initialization."""
    print(f"Python version: {platform.python_version()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    verify_project_directories()
    print("Stage 1 project initialization successful.")


if __name__ == "__main__":
    main()

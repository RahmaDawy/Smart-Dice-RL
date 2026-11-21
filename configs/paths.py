"""Central place for all file-system paths (pathlib, Windows friendly)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_DIR = RESULTS_DIR / "csv"


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (MODELS_DIR, PLOTS_DIR, CSV_DIR):
        folder.mkdir(parents=True, exist_ok=True)

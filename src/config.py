from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

RANDOM_STATE = 42
SAMPLE_FRAC = 1.0  # Use 0.20 for a quick trial; use 1.0 for final experiments.

for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

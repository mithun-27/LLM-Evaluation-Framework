from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = BASE_DIR / "datasets"

REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR = BASE_DIR / "models"

EXPERIMENT_DIR = BASE_DIR / "experiments"
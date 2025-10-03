import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Displays info messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Logger - logging.getLogger(__name__) reflects the module
logger: logging.Logger = logging.getLogger("config")

BASE_DIR: Path = Path(os.getenv("BASE_DIR", "."))
RESULT_FIGURES_DATA_PATH: Path = BASE_DIR / "result_figs"
SRC_PATH: Path = BASE_DIR / "src"

DATA_PATH: Path = BASE_DIR / "data"
PREFILTERED_DATA_PATH: Path = DATA_PATH / "stand_knn_features_variance1000.tsv"
CLINICAL_DATA_PATH: Path = DATA_PATH / "data_clinical_patient.txt"
paths: list[Path] = [
    BASE_DIR,
    DATA_PATH,
    SRC_PATH,
    PREFILTERED_DATA_PATH,
    CLINICAL_DATA_PATH,
    RESULT_FIGURES_DATA_PATH,
]

# Printing config overview
logging.info(
    "\nDEFINED PROJECT PATHS:\n" + "\n".join([str(path) for path in paths])
)

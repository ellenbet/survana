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
PREFILTERED_DATA_PATH: Path = DATA_PATH / "example_name.tsv"
CLINICAL_DATA_PATH: Path = DATA_PATH / "clinical_example_name.txt"
paths: list[Path] = [
    BASE_DIR,
    DATA_PATH,
    SRC_PATH,
    PREFILTERED_DATA_PATH,
    CLINICAL_DATA_PATH,
    RESULT_FIGURES_DATA_PATH,
]

CENSOR_STATUS: str = "EXAMPLE_CENSOR_STATUS"
MONTHS_BEFORE_EVENT: str = "EXEMPLE_MONTHS"
P_ID = "EXAMPLE_PATIENT_ID"

COXPH_EXPERIMENT_ID = "EXAMPLE_EXPERIMENT_ID"
COXPH_NON_NESTED_EXPERIMENT_ID = "EXAMPLE_NON_NESTED_EXPERIMENT_ID"

# Tuning and trial parameters
SKF_SPLITS = 5  # Stratified K-fold splits
RSKF_SPLITS = 5  # Repeated stratified K-fold splits (Equals K)
RSKF_REPEATS = 10  # Number of repeats per fold
MCCV_SPLITS = 10  # Monte Carlo CV splits
COEF_ZERO_CUTOFF = 4  # Coef is zero if 10log(coef) <= 4
N_TRIALS = 100  # Optuna trials
LOG_LAMBDA_MIN, LOG_LAMBDA_MAX = -8, 1
N_LAMBDA = (
    100  # Total numbers of lambda (hyperparam in Ridge/Lasso regression)
)

# Model parameters
MODEL_ITERATIONS = 100
MODEL_TYPE = "lasso"

# Printing config overview
logging.info(
    "\nDEFINED PROJECT PATHS:\n"
    + "\n".join([str(path) for path in paths])
    + "\nStarting for MODEL_TYPE: {MODEL_TYPE}"
)

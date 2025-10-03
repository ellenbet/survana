import logging

# import warnings
from typing import Any

# import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sksurv.linear_model as lm
from sklearn.model_selection import train_test_split

from config import PREFILTERED_DATA_PATH
from data_processing.dataloaders import load_data_for_sksurv_coxnet

# from sklearn.exceptions import FitFailedWarning
# from sklearn.model_selection import GridSearchCV, KFold, train_test_split
# from sklearn.pipeline import Pipeline, make_pipeline

# from data_processing.datamergers import merge_features_with_clinical_data

SEED: int = 42
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger("CoxNet")

logger.info(f"\n\nRetrieving data from path {PREFILTERED_DATA_PATH}")

CENSOR_STATUS: str = "RFS_STATUS"
MONTHS_BEFORE_EVENT: str = "RFS_MONTHS"
P_ID = "PATIENT_ID"

data_collection: tuple[
    pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
] = load_data_for_sksurv_coxnet(
    str(PREFILTERED_DATA_PATH),
    response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
)


data: pd.DataFrame = data_collection[0]
X: pd.DataFrame = data_collection[1]
y: np.recarray[tuple[Any, ...], np.dtype[Any]] = data_collection[2]


logger.info(
    "\nCensored patients in data: "
    + str(len(data[CENSOR_STATUS]) - data[CENSOR_STATUS].sum())
    + " out of "
    + str(len(data[CENSOR_STATUS])),
)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)


def objective(trial) -> float | Any:
    params = {"lambda": trial.suggest_float("lambda", 1e-5, 1)}
    model = lm.CoxPHSurvivalAnalysis(
        verbose=True,
        alpha=params["lambda"],
        n_iter=100,
    )
    # NUM_TRIALS = 30
    # non_nested_scores = np.zeros(NUM_TRIALS)
    # nested_scores = np.zeros(NUM_TRIALS)
    # inner_cv = KFold(n_splits=4, shuffle=True)
    # outer_cv = KFold(n_splits=4, shuffle=True)
    model.fit(x_train, y_train)
    score: float | Any = model.score(x_test, y_test)
    return score


study: optuna.Study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
"""

merged_df: pd.DataFrame = merge_features_with_clinical_data(
    str(PREFILTERED_DATA_PATH),
    str(CLINICAL_DATA_PATH),
    merge_on=P_ID,
)


X[P_ID] = data[P_ID]
assert (
    X[P_ID].all() == merged_df[P_ID].all()
), "not able to sort X by merged due to mismatch of patient id order"
"""

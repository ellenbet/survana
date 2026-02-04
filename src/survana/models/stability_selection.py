import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sksurv.linear_model as lm
import tqdm
from result_processing.plotting import Plotter
from result_processing.result import Result

from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.config import CONFIG, PATHS
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet
from survana.tuning.training_wrappers import robust_train

CENSOR_STATUS: str = CONFIG["columns"]["censor_status"]
COXPH_EXPERIMENT_ID: str = CONFIG["experiments"]["coxph_experiment_id"]
COXPH_NON_NESTED_EXPERIMENT_ID = CONFIG["experiments"][
    "coxph_non_nested_experiment_id"
]
MONTHS_BEFORE_EVENT: str = CONFIG["columns"]["months_before_event"]
N_TRIALS: int = CONFIG["tuning"]["n_trials"]
RSKF_REPEATS: int = CONFIG["tuning"]["rskf_repeats"]
RSKF_SPLITS: int = CONFIG["tuning"]["rskf_splits"]
SKF_SPLITS: int = CONFIG["tuning"]["skf_splits"]
COEF_ZERO_CUTOFF: int = CONFIG["tuning"]["coef_zero_cutoff"]
LOG_LAMBDA_MAX: int = CONFIG["tuning"]["log_lambda_max"]
LOG_LAMBDA_MIN: int = CONFIG["tuning"]["log_lambda_min"]
MODEL_TYPE: str = CONFIG["model"]["model_type"]
N_LAMBDA: int = CONFIG["tuning"]["n_lambda"]
PREFILTERED_DATA_PATH: Path = PATHS["PREFILTERED_DATA_PATH"]
RESULT_FIGURES_DATA_PATH: Path = PATHS["RESULT_FIGURES_DATA_PATH"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def stability_selection():
    """Stability selection function with subsampling B * N_LAMBDA times.
    Number of sumsamples per lambda is B = RSKF_SPLITS * RSKF_REPEATS,
    see config.py for constant definitions.

    Function relies on robust_train() function which can be used
    with Ridge and Lasso Cox-regression, can be written to function
    with Elastic Net as well.

    """

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    subsampler: Subsampler = Subsampler.repeated_kfold(
        n_splits=RSKF_SPLITS, n_repeats=RSKF_REPEATS
    )

    sksurv_data: SksurvData = SksurvData(data_collection=data_collection)
    feature_no: int = sksurv_data.X.shape[1]
    art_X: np.ndarray[tuple[Any, ...], np.dtype[Any]] = ArtificialGenerator(
        feature_no, ArtificialType.KNOCKOFF
    ).fit_transform(sksurv_data.X)

    feature_names: list[str] = list(sksurv_data.X.columns)
    artificial_names: list[str] = [
        f"artificial_{i}" for i in range(feature_no)
    ]

    logger.info(
        "starting stability selectiont trial with"
        + f"\nLOG_LAMBDA_MIN: {LOG_LAMBDA_MIN}"
        + f"\nLOG_LAMBDA_MAX: {LOG_LAMBDA_MAX}"
        + f"\nN_LAMBDA: {N_LAMBDA}"
    )
    params: np.ndarray = np.logspace(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX, N_LAMBDA)
    results: Result = Result(
        feature_names + artificial_names,
        rounding_cutoff=COEF_ZERO_CUTOFF,
        bin_min=LOG_LAMBDA_MIN,
        bin_max=LOG_LAMBDA_MAX,
    )
    for test, train in subsampler.split(art_X, sksurv_data.y):
        for param in tqdm.tqdm(params):
            logger.info(f"Fitting model for hyperparam {param}")
            model: (
                lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float
            ) = robust_train(MODEL_TYPE, art_X, sksurv_data.y, param, train)

            if isinstance(model, float):
                pass
            else:
                score: float | Any = model.score(
                    art_X[test, :], y=sksurv_data.y[test]
                )
                results.save_results(score, param, model.coef_.flatten())
                logger.info(f"param: {param} with score: " + f"{score}")

        results.save_results_to_file()
    plotter: Plotter = Plotter(results.get_result_path())
    plotter.plot_stability_path(save=True)
    plotter.plot_stability_path_with_thresh(save=True)

import logging
from functools import partial
from typing import Any

import numpy as np
import optuna
import pandas as pd
import sksurv.linear_model as lm
import tqdm

from config import CENSOR_STATUS, MONTHS_BEFORE_EVENT, PREFILTERED_DATA_PATH
from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet
from survana.data_processing.result_models import Result
from survana.tuning.optuna_objectives import (
    categorical_stability_objective,
    stability_objective,
)
from survana.tuning.training_wrappers import robust_train

SKF_SPLITS = 2
RSKF_SPLITS = 2
RSKF_REPEATS = 1
N_TRIALS = 1

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def subsampled_stability_coxph():
    """_summary_"""

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    # B = 50 model training sessions for each hyperparameter lambda
    subsampler: Subsampler = Subsampler.repeated_kfold(
        n_splits=5, n_repeats=10
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
    start_lambda = -8
    stop_lambda = 1
    N_LAMBDA = 100
    params: np.ndarray = np.logspace(start_lambda, stop_lambda, N_LAMBDA)
    results: Result = Result(
        feature_names + artificial_names,
        rounding_cutoff=4,
        bin_min=start_lambda,
        bin_max=stop_lambda,
    )
    for test, train in subsampler.split(art_X, sksurv_data.y):
        for param in tqdm.tqdm(params):
            logger.info(f"Fitting model for hyperparam {param}")
            model: (
                lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float
            ) = robust_train("lasso", art_X, sksurv_data.y, param, train)

            if isinstance(model, float):
                pass
            else:
                results.save_results(param, model.coef_.flatten())
                logger.info(
                    f"param: {param} with score: "
                    + f"{model.score(art_X[test, :], y=sksurv_data.y[test])}"
                )
    results.plot_results()
    # results.get_results_file()


def subsampled_stability_coxph_optuna():
    """_summary_"""

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    # B = 50 model training sessions for each hyperparameter lambda
    subsampler: Subsampler = Subsampler.repeated_kfold(
        n_splits=5, n_repeats=10
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
    results: Result = Result(feature_names + artificial_names)

    study: optuna.Study = optuna.create_study(direction="maximize")
    for test, train in subsampler.split(art_X, sksurv_data.y):
        wrapped_objective: partial[float | Any] = partial(
            stability_objective,
            X=art_X,
            y=sksurv_data.y,
            results=results,
            test_ind=test,
            train_ind=train,
        )
        study.optimize(wrapped_objective, n_trials=2, show_progress_bar=True)

    results.plot_results()
    results.get_results_file()


def categorical_subsampled_stability_coxph_optuna():
    """_summary_"""

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    # B = 50 model training sessions for each hyperparameter lambda
    subsampler: Subsampler = Subsampler.repeated_kfold(
        n_splits=5, n_repeats=10
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
    results: Result = Result(feature_names + artificial_names)

    study: optuna.Study = optuna.create_study(direction="maximize")
    for test, train in subsampler.split(art_X, sksurv_data.y):
        wrapped_objective: partial[float | Any] = partial(
            categorical_stability_objective,
            X=art_X,
            y=sksurv_data.y,
            results=results,
            test_ind=test,
            train_ind=train,
        )
        study.optimize(wrapped_objective, n_trials=2, show_progress_bar=True)

    results.plot_results()
    results.get_results_file()

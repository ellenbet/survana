import logging
from typing import Any, Iterator

import mlflow
import numpy as np
import pandas as pd
import sksurv.linear_model as lm

from config import LOG_LAMBDA_MAX, LOG_LAMBDA_MIN, MODEL_TYPE
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.tuning.training_wrappers import robust_train

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

# TODO clean up optuna objectives


def sksurv_objective_with_args(
    trial,
    sksurv_data,
    outer_train_ind: list[int],
) -> float | Any:
    """Optuna objective, uses stratified_repeated_kfold_splits

    Args:
        trial (_type_): _description_
        sksurv_data (_type_): _description_
        outer_train_ind (_type_): _description_
        rskf_repeats (_type_): _description_
        rskf_splits (_type_): _description_

    Returns:
        float | Any: _description_
    """
    params = {
        "lambda": trial.suggest_float("lambda", LOG_LAMBDA_MIN, LOG_LAMBDA_MAX)
    }
    nested_scores: list[float] = []
    model = lm.CoxPHSurvivalAnalysis(
        verbose=True,
        alpha=params["lambda"],
        n_iter=100,
    )

    inner_X: pd.DataFrame = sksurv_data.X.iloc[outer_train_ind]
    inner_y = sksurv_data.y[outer_train_ind]

    inner_split: Iterator[Any] = sksurv_data.stratified_repeated_kfold_splits(
        X=inner_X, y=inner_y
    )
    for _, (inner_train_ind, inner_test_ind) in enumerate(inner_split):
        model.fit(inner_X.iloc[inner_train_ind], y=inner_y[inner_train_ind])
        inner_score: float | Any = model.score(
            inner_X.iloc[inner_test_ind], y=inner_y[inner_test_ind]
        )
        nested_scores.append(inner_score)

    score: np.floating[Any] = np.mean(nested_scores)
    return score


def mlflow_sksurv_objective_with_args(
    trial,
    sksurv_data: SksurvData,
    outer_train_ind: int,
    rskf_repeats: int,
    rskf_splits: int,
    run_name: str,
    experiment_id: str,
) -> float | Any:
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=True
    ):
        params: dict[str, Any] = {
            "lambda": trial.suggest_float(
                "lambda", LOG_LAMBDA_MIN, LOG_LAMBDA_MAX
            )
        }

        nested_scores: list[float] = []
        model = lm.CoxPHSurvivalAnalysis(
            verbose=True,
            alpha=params["lambda"],
            n_iter=100,
        )

        inner_X: pd.Series = sksurv_data.X.iloc[outer_train_ind]
        inner_y: np.recarray = sksurv_data.y[outer_train_ind]
        subsampler: Subsampler = Subsampler.repeated_kfold(
            n_splits=rskf_splits, n_repeats=rskf_repeats
        )
        inner_split: Iterator[Any] = subsampler.split(X=inner_X, y=inner_y)
        for inner_fold, (inner_train_ind, inner_test_ind) in enumerate(
            inner_split
        ):
            try:
                model.fit(
                    inner_X.iloc[inner_train_ind], y=inner_y[inner_train_ind]
                )
                inner_score: float | Any = model.score(
                    inner_X.iloc[inner_test_ind], y=inner_y[inner_test_ind]
                )
                nested_scores.append(inner_score)
            except np.linalg.LinAlgError:
                logger.error(
                    "Singular matrix multiplication attempted"
                    + f"skipping results for inner fold no. {inner_fold}"
                )
                pass

        score: float = float(np.mean(nested_scores))
        mlflow.log_param("lambda", params["lambda"])
        mlflow.log_metric("child c-index", score)
        mlflow.log_metric(
            "child c-index variance", float(np.var(nested_scores))
        )

    return score


def mlflow_non_nested_objective_with_args(
    trial,
    sksurv_data: SksurvData,
    test_ind: list[int],
    train_ind: list[int],
    run_name: str,
    experiment_id: str,
) -> float | Any:
    """_summary_

    Args:
        trial (_type_): _description_
        sksurv_data (SksurvData): _description_
        test_ind (list[int]): _description_
        train_ind (list[int]): _description_
        run_name (str): _description_
        experiment_id (str): _description_

    Returns:
        float | Any: _description_
    """
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=False
    ):
        params: dict[str, Any] = {
            "lambda": trial.suggest_float(
                "lambda", LOG_LAMBDA_MIN, LOG_LAMBDA_MAX, log=True
            )
        }

        model: lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float = (
            robust_train(
                model_type=MODEL_TYPE,
                X=sksurv_data.X.to_numpy(),
                y=sksurv_data.y,
                param=params["lambda"],
                train_ind=train_ind,
            )
        )

        score: float | Any = (
            model
            if isinstance(model, float)
            else model.score(
                sksurv_data.X[test_ind, :], y=sksurv_data.y[test_ind]
            )
        )

        mlflow.log_param("lambda", params["lambda"])
        mlflow.log_metric("c-index", score)
        return score


def mlflow_non_nested_objective_artificial(
    trial,
    X: np.ndarray,
    y: np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
    test_ind: list[int],
    train_ind: list[int],
    run_name: str,
    experiment_id: str,
) -> float | Any:
    """_summary_

    Args:
        X (np.ndarray): _descrption_
        y (np.ndarray): _descprition_
        trial (_type_): _description_
        test_ind (list[int]): _description_
        train_ind (list[int]): _description_
        run_name (str): _description_
        experiment_id (str): _description_

    Returns:
        float | Any: _description_
    """
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=False
    ):
        params: dict[str, Any] = {
            "lambda": trial.suggest_float(
                "lambda", LOG_LAMBDA_MIN, LOG_LAMBDA_MAX, log=True
            )
        }

        model: lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float = (
            robust_train(
                model_type=MODEL_TYPE,
                X=X,
                y=y,
                param=params["lambda"],
                train_ind=train_ind,
            )
        )

        score: float | Any = (
            model
            if isinstance(model, float)
            else model.score(X[test_ind, :], y=y[test_ind])
        )

        mlflow.log_param("lambda", params["lambda"])
        mlflow.log_metric("c-index", score)
        return score

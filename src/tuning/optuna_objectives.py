from typing import Any, Iterator

import mlflow
import numpy as np
import pandas as pd
import sksurv.linear_model as lm

from data_processing.data_models import SksurvData


def sksurv_objective_with_args(
    trial,
    sksurv_data,
    outer_train_ind,
    rskf_repeats,
    rskf_splits,
) -> float | Any:
    params = {"lambda": trial.suggest_float("lambda", 1e-5, 1)}
    nested_scores: list[float] = []
    model = lm.CoxPHSurvivalAnalysis(
        verbose=True,
        alpha=params["lambda"],
        n_iter=100,
    )

    inner_X: pd.DataFrame = sksurv_data.X.iloc[outer_train_ind]
    inner_y = sksurv_data.y[outer_train_ind]

    inner_split: Iterator[Any] = sksurv_data.stratified_repeated_kfold_splits(
        X=inner_X,
        y=inner_y,
        n_repeats=rskf_repeats,
        n_splits=rskf_splits,
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
            "lambda": trial.suggest_float("lambda", 1e-5, 1)
        }

        nested_scores: list[float] = []
        model = lm.CoxPHSurvivalAnalysis(
            verbose=True,
            alpha=params["lambda"],
            n_iter=100,
        )

        inner_X: pd.Series = sksurv_data.X.iloc[outer_train_ind]
        inner_y: np.recarray = sksurv_data.y[outer_train_ind]

        inner_split: Iterator[
            Any
        ] = sksurv_data.stratified_repeated_kfold_splits(
            X=inner_X,
            y=inner_y,
            n_repeats=rskf_repeats,
            n_splits=rskf_splits,
        )
        for _, (inner_train_ind, inner_test_ind) in enumerate(inner_split):
            model.fit(
                inner_X.iloc[inner_train_ind], y=inner_y[inner_train_ind]
            )
            inner_score: float | Any = model.score(
                inner_X.iloc[inner_test_ind], y=inner_y[inner_test_ind]
            )
            nested_scores.append(inner_score)

        score: float = float(np.mean(nested_scores))
        mlflow.log_param("lambda", params["lambda"])
        mlflow.log_metric("c-index", score)
        mlflow.log_metric("c-index variance", float(np.var(nested_scores)))

    return score

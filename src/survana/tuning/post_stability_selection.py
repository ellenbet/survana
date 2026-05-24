from functools import partial
from typing import Any

import numpy as np
import optuna
import sksurv.linear_model as lm
from numpy import floating
from optuna import Study
from tqdm import tqdm

from survana.config import CONFIG
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.tuning.optuna_objectives import (
    cox_cv_objective,
    random_survival_forest_objective,
)
from survana.tuning.training_wrappers import robust_train

# optuna.logging.set_verbosity(optuna.logging.WARNING)


def cox_final_tuning(
    data: SksurvData,
    model_type="ridge",
    n_alphas: int = 100,
    min_alpha_log=-8,
    max_alpha_log=1,
    splits: int = 5,
    repeats: int = 1,
) -> dict[str, dict[str, Any] | np.ndarray]:
    """Final tuner which uses gridsearch to find best alpha, search is
    spread across logspace.

    Args:
        data (SksurvData): the full data
        n_alphas (int, optional): number of alphas. Defaults to 100.
        min_alpha_log (int, optional): lowest log(alpha). Defaults to -10.
        max_alpha_log (int, optional): highest log(alpha). Defaults to 2.
        n_iter (int, optional): model iterations during fit. Defaults to 100.
        splits (int, optional): kfold splits. Defaults to 5.
        repeats (int, optional): repeated kfolds to decrease stochastic
        results. Defaults to 2.

    Returns:
        dict[str, Any]: dict with params, scores and std_dev keys.
    """
    top_score: list[float] = []
    stds: list[float] = []
    params: dict[str, np.ndarray] = {
        "alpha": np.logspace(min_alpha_log, max_alpha_log, n_alphas)
    }
    for param in tqdm(params["alpha"]):
        scores: list[float] = []

        subsampler: Subsampler = Subsampler.repeated_kfold(
            n_splits=splits, n_repeats=repeats
        )

        for train_ind, test_ind in subsampler.split(X=data.X, y=data.y):
            trained_model: (
                lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float
            ) = robust_train(
                model_type=model_type,
                X=data.X.to_numpy(),
                y=data.y,
                train_ind=train_ind,
                param=param,
            )
            if hasattr(trained_model, "score"):
                score = trained_model.score(
                    data.X.iloc[test_ind, :], data.y[test_ind]
                )
                scores.append(score)
            else:
                score = float(trained_model)
        top_score.append(np.mean(scores))
        stds.append(np.std(scores))
    return {
        "params": params,
        "scores": np.array(top_score),
        "stds": np.array(stds),
    }


def coxph_final_tuning_optuna(
    data: SksurvData,
    min_alpha_log: int = CONFIG["tuning"]["log_lambda_min"],
    max_alpha_log=CONFIG["tuning"]["log_lambda_max"],
    n_trials: int = 20,
    n_splits: int = CONFIG["tuning"]["rskf_splits"],
    n_repeats: int = CONFIG["tuning"]["rskf_repeats"],
    model_type: str = "ridge",
) -> dict[str, dict[str, Any] | np.ndarray]:
    """Twin function to cox_final_tuning above but using
    Optuna instead to compare results and efficiency. Returns
    hyperparameters with model score (c-index) and std.

    Args:
        data (SksurvData): full data
        min_alpha_log (int, optional): minimum log(alpha). Defaults to -10.
        max_alpha_log (int, optional): maximum log(alpha). Defaults to 2.
        n_trials (int, optional): number of optuna trials. Defaults to 50.

    Returns:
        dict[str, Any]: dict with params, scores and stds keys.
    """
    study = optuna.create_study(direction="maximize")
    wrapped: partial[floating] = partial(
        cox_cv_objective,
        X=data.X,
        y=data.y,
        min_alpha_log=min_alpha_log,
        max_alpha_log=max_alpha_log,
        n_splits=n_splits,
        n_repeats=n_repeats,
        model_type=model_type,
    )
    study.optimize(wrapped, n_trials=n_trials)  # type: ignore
    values: np.ndarray = np.array([trial.value for trial in study.trials])
    stds: np.ndarray = np.array(
        [trial.user_attrs["std_dev"] for trial in study.trials]
    )
    alphas: list[Any] = [trial.params["alpha"] for trial in study.trials]
    params: dict[str, Any] = {}
    params["alpha"] = alphas
    return {"params": params, "scores": values, "stds": stds}


def rsf_final_tuning(data: SksurvData, n_trials: int = 20) -> dict[str, Any]:
    """Random forest final tuning using optuna. Depends on the
    random_survival_forest_objective in the tuning module, uses
    CV inside objective to be tuned.

    Args:
        data (SksurvData): full data.
        n_trials (int, optional): number of optuna trials. Defaults to 20.

    Returns:
        dict[str, Any]: dict with params, scores and std_dev keys.
    """
    study: Study = optuna.create_study(direction="maximize")
    wrapped: partial[floating[Any]] = partial(
        random_survival_forest_objective, X=data.X, y=data.y
    )
    study.optimize(wrapped, n_trials=n_trials, show_progress_bar=True)
    values = np.array([trial.value for trial in study.trials])
    stds = np.array([trial.user_attrs["std_dev"] for trial in study.trials])
    params = [trial.params for trial in study.trials]
    return {"params": params, "scores": values, "stds": stds}


def cox(sksurv: SksurvData, n_rep=10, n_spl=5) -> dict[str, list[float]]:
    splitter: Subsampler = Subsampler.repeated_kfold(
        n_repeats=n_rep, n_splits=n_spl
    )
    scores: list[float] = []
    for train, test in splitter.split(sksurv.X, sksurv.y):
        X_train = sksurv.X.iloc[train, :]
        y_train = sksurv.y[train]
        model = lm.CoxPHSurvivalAnalysis(alpha=0, n_iter=200)

        try:
            model.fit(X_train, y_train)
        except np.linalg.LinAlgError:
            print(
                "\nSingular matrix multiplication attempted - "
                + "skipping results..",
            )
            continue
        score = model.score(sksurv.X.iloc[test, :], sksurv.y[test])
        scores.append(score)
    return {"scores": scores}


def cox_return_top_model(sksurv: SksurvData, n_rep=10, n_spl=5):
    splitter: Subsampler = Subsampler.repeated_kfold(
        n_repeats=n_rep, n_splits=n_spl
    )
    scores: list[float] = []
    top_score = 0
    top_model = None
    for train, test in splitter.split(sksurv.X, sksurv.y):
        X_train = sksurv.X.iloc[train, :]
        y_train = sksurv.y[train]
        model = lm.CoxPHSurvivalAnalysis(alpha=0, n_iter=200)

        try:
            model.fit(X_train, y_train)
        except np.linalg.LinAlgError:
            print(
                "\nSingular matrix multiplication attempted - "
                + "skipping results..",
            )
        score = model.score(sksurv.X.iloc[test, :], sksurv.y[test])
        if score > top_score:
            top_score = score
            top_model = model
        scores.append(score)
    return top_model

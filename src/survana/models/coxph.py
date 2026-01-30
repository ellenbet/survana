import logging
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import sksurv.linear_model as lm
from matplotlib import pyplot as plt
from mlflow.entities.experiment import Experiment

from survana.config import CONFIG, PATHS
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet
from survana.tuning.optuna_objectives import (
    mlflow_non_nested_objective_with_args,
    mlflow_sksurv_objective_with_args,
)

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
PREFILTERED_DATA_PATH: Path = PATHS["PREFILTERED_DATA_PATH"]
RESULT_FIGURES_DATA_PATH: Path = PATHS["RESULT_FIGURES_DATA_PATH"]

today: date = date.today()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def coxph() -> None:
    """Full model with Sksurv Cox-Lasso, using Optuna and MLFlow"""
    current_datetime: datetime = datetime.now()
    current_timestamp: float = current_datetime.timestamp()
    logger.info(f"\n\nRetrieving data from path {PREFILTERED_DATA_PATH}")

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    sksurv_data: SksurvData = SksurvData(data_collection=data_collection)

    logger.info(
        "\nCensored patients in data: "
        + str(round(sksurv_data.censored_patient_percentage, 2))
    )

    experiment: Experiment | None = mlflow.get_experiment_by_name(
        COXPH_EXPERIMENT_ID
    )
    assert (
        experiment is not None
    ), f"no data found on experiment {COXPH_EXPERIMENT_ID}"
    ident = experiment.experiment_id
    logger.info(f"\nStarting MLflow with experiment id {ident}")

    outer_fold: int = 0
    # parent run
    with mlflow.start_run(
        experiment_id=ident,
        run_name=str(today)
        + "_"
        + COXPH_EXPERIMENT_ID
        + f"_parent_{outer_fold}"
        + str(current_timestamp)[:5],
    ):
        subsampler: Subsampler = Subsampler.kfold(n_splits=SKF_SPLITS)
        outer_best_param: list[float] = [0.0] * SKF_SPLITS
        outer_best_score: list[float] = [0.0] * SKF_SPLITS
        for outer_fold, (outer_train_ind, outer_test_ind) in enumerate(
            subsampler.split(sksurv_data.X, sksurv_data.y)
        ):
            wrapped_objective: partial[float | Any] = partial(
                mlflow_sksurv_objective_with_args,
                sksurv_data=sksurv_data,
                outer_train_ind=outer_train_ind,
                rskf_repeats=RSKF_REPEATS,
                rskf_splits=RSKF_SPLITS,
                run_name=str(today) + "_child_outer_fold_" + str(outer_fold),
                experiment_id=ident,
            )
            study: optuna.Study = optuna.create_study(direction="maximize")
            study.optimize(
                wrapped_objective, n_trials=N_TRIALS, show_progress_bar=True
            )
            best_param = study.best_params["lambda"]
            results = study.trials_dataframe()
            results["value"].sort_values().reset_index(drop=True).plot(
                label=f"Fold: {outer_fold}"
            )
            plt.title("Convergence plot, Nested stratified repeated K-Fold")
            plt.xlabel("Iteration")
            plt.ylabel("C-index")

            outer_best_param[outer_fold] = best_param
            model = lm.CoxPHSurvivalAnalysis(
                verbose=True, alpha=best_param, n_iter=100
            )
            try:
                model.fit(
                    sksurv_data.X.iloc[outer_train_ind],
                    y=sksurv_data.y[outer_train_ind],
                )
                outer_score: float | Any = model.score(
                    sksurv_data.X.iloc[outer_test_ind],
                    y=sksurv_data.y[outer_test_ind],
                )
                outer_best_score[outer_fold] = outer_score
            except np.linalg.LinAlgError:
                logger.error(
                    "Singular matrix multiplication attempted"
                    + f"skipping results for outer fold no. {outer_fold}"
                )
                pass

        outer_best_score = (
            [0] if sum(outer_best_score) == 0 else outer_best_score
        )
        best: float = max(outer_best_score)
        mlflow.log_metric("parent best c-index", best)
        mlflow.log_param(
            "parent best lambda",
            outer_best_param[outer_best_score.index(best)],
        )
        mlflow.log_metric("parent mean c-index", float(np.mean(outer_score)))

        plt.legend()
        plt.show()

        filename: str = (
            str(RESULT_FIGURES_DATA_PATH)
            + "/coxph_convergence_plot_optuna_"
            + str(today)
        )
        plt.savefig(filename)
        mlflow.log_artifact(filename + ".png")
        plt.close()
        logger.info(
            f"Best score found at {best} with lambda "
            + f"{outer_best_param[outer_best_score.index(best)]}"
        )
        # ending run to start new logging session for next cv
        mlflow.end_run()


def non_nested_coxph() -> None:
    """Full model with Sksurv Cox-Lasso, using Optuna and MLFlow,
    but non-nested"""

    logger.info(f"\n\nRetrieving data from path {PREFILTERED_DATA_PATH}")

    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet(
        str(PREFILTERED_DATA_PATH),
        response_variables=(CENSOR_STATUS, MONTHS_BEFORE_EVENT),
    )

    sksurv_data: SksurvData = SksurvData(data_collection=data_collection)
    subsampler = Subsampler.shuffle_split(n_splits=5)

    logger.info(
        "\nCensored patients in data: "
        + str(round(sksurv_data.censored_patient_percentage, 2))
    )

    experiment: Experiment | None = mlflow.get_experiment_by_name(
        COXPH_NON_NESTED_EXPERIMENT_ID
    )
    assert (
        experiment is not None
    ), f"no data found on experiment {COXPH_NON_NESTED_EXPERIMENT_ID}"
    ident = experiment.experiment_id
    logger.info(f"\nStarting MLflow with experiment id {ident}")

    for test, train in subsampler.split(sksurv_data.X, sksurv_data.y):
        wrapped_objective: partial[float | Any] = partial(
            mlflow_non_nested_objective_with_args,
            sksurv_data=sksurv_data,
            test_ind=test,
            train_ind=train,
            run_name=str(today) + "_tester",
            experiment_id=ident,
        )
        study: optuna.Study = optuna.create_study(direction="maximize")
        study.optimize(wrapped_objective, n_trials=20)
        best_param = study.best_params["lambda"]
        logger.info(f"Best lambda is {best_param}")

        # ending run to start new logging session for next cv
        mlflow.end_run()

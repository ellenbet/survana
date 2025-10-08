import logging
from datetime import date, datetime
from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import sksurv.linear_model as lm
from matplotlib import pyplot as plt
from mlflow.entities.experiment import Experiment

from config import (
    CENSOR_STATUS,
    COXPH_EXPERIMENT_ID,
    MONTHS_BEFORE_EVENT,
    PREFILTERED_DATA_PATH,
    RESULT_FIGURES_DATA_PATH,
)
from data_processing.data_models import SksurvData
from data_processing.dataloaders import load_data_for_sksurv_coxnet
from tuning.optuna_objectives import mlflow_sksurv_objective_with_args

SKF_SPLITS = 5
RSKF_SPLITS = 5
RSKF_REPEATS = 5
N_TRIALS = 10


def coxph() -> None:
    today: date = date.today()
    current_datetime: datetime = datetime.now()
    current_timestamp: float = current_datetime.timestamp()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger: logging.Logger = logging.getLogger(__name__)

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

        outer_best_param: list[float] = [0.0] * SKF_SPLITS
        outer_best_score: list[float] = [0.0] * SKF_SPLITS
        for outer_fold, (outer_train_ind, outer_test_ind) in enumerate(
            sksurv_data.stratified_kfold_splits(n_splits=SKF_SPLITS)
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
            study.optimize(wrapped_objective, n_trials=N_TRIALS)
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

import logging
from datetime import date
from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import sksurv.linear_model as lm
from matplotlib import pyplot as plt

from config import (
    CENSOR_STATUS,
    MONTHS_BEFORE_EVENT,
    PREFILTERED_DATA_PATH,
    RESULT_FIGURES_DATA_PATH,
)
from data_processing.data_models import SksurvData
from data_processing.dataloaders import load_data_for_sksurv_coxnet
from tuning.optuna_objectives import sksurv_objective_with_args

# from data_processing.datamergers import merge_features_with_clinical_data


def coxph() -> None:
    today = date.today()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger: logging.Logger = logging.getLogger("CoxNet")

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

    # parent run
    with mlflow.start_run(
        experiment_id=mlflow.create_experiment("coxph"), nested=True
    ):
        SKF_SPLITS = 5
        RSKF_SPLITS = 5
        RSKF_REPEATS = 3
        N_TRIALS = 5

        outer_best_param: list[float] = [0.0] * SKF_SPLITS
        outer_best_score: list[float] = [0.0] * SKF_SPLITS
        for outer_fold, (outer_train_ind, outer_test_ind) in enumerate(
            sksurv_data.stratified_kfold_splits(n_splits=SKF_SPLITS)
        ):
            wrapped_objective: partial[float | Any] = partial(
                sksurv_objective_with_args,
                sksurv_data=sksurv_data,
                outer_train_ind=outer_train_ind,
                rskf_repeats=RSKF_REPEATS,
                rskf_splits=RSKF_SPLITS,
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
            model.fit(
                sksurv_data.X.iloc[outer_train_ind],
                y=sksurv_data.y[outer_train_ind],
            )
            outer_score: float | Any = model.score(
                sksurv_data.X.iloc[outer_test_ind],
                y=sksurv_data.y[outer_test_ind],
            )
            outer_best_score[outer_fold] = outer_score
            mlflow.log_metric("outer cv best c-index", study.best_value)
            mlflow.log_params(study.best_params)

        best = max(outer_best_score)
        mlflow.log_metric("total best c-index", best)
        mlflow.log_param(
            "total lambda", outer_best_param[outer_best_score.index(best)]
        )
        plt.legend()
        plt.show()
        plt.savefig(
            str(RESULT_FIGURES_DATA_PATH)
            + "/coxph_convergence_plot_optuna_"
            + str(today)
        )
        logger.info(
            f"Best score found at {best} with lambda"
            + f"{outer_best_param[outer_best_score.index(best)]}"
        )

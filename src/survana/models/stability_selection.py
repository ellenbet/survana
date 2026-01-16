import logging
from typing import Any

import numpy as np
import pandas as pd
import sksurv.linear_model as lm
import tqdm

from config import (
    CENSOR_STATUS,
    COEF_ZERO_CUTOFF,
    LOG_LAMBDA_MAX,
    LOG_LAMBDA_MIN,
    MONTHS_BEFORE_EVENT,
    N_LAMBDA,
    PREFILTERED_DATA_PATH,
    RSKF_REPEATS,
    RSKF_SPLITS,
)
from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet
from survana.data_processing.result_models import Result
from survana.tuning.training_wrappers import robust_train

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def subsampled_stability_coxph():
    """Stability selection function
    Number of sumsamples per lambda is B = RSKF_SPLITS * RSKF_REPEATS,
    see config.py for constant definitions.
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
            ) = robust_train("lasso", art_X, sksurv_data.y, param, train)

            if isinstance(model, float):
                pass
            else:
                results.save_results(param, model.coef_.flatten())
                logger.info(
                    f"param: {param} with score: "
                    + f"{model.score(art_X[test, :], y=sksurv_data.y[test])}"
                )
        results.get_results_file()
    results.plot_results()

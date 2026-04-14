import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import sksurv.linear_model as lm
import tqdm
from scipy.sparse.linalg import ArpackNoConvergence
from sklearn.exceptions import ConvergenceWarning

from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.config import CONFIG
from survana.data_processing.data_models import SksurvData
from survana.data_processing.data_subsampler import Subsampler
from survana.result_processing.result import Result
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)
from survana.tuning.training_wrappers import robust_train

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


class StabilitySelectionCancelled(Exception):
    pass


def stability_selection(
    data_collection: tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.recarray[tuple[Any, ...], np.dtype[Any]],
    ],
    plot: bool = False,
    stop_requested: Callable[[], bool] | None = None,
) -> SingleStabilityResult:
    """Stability selection function with subsampling B * N_LAMBDA times.
    Number of sumsamples per lambda is B = RSKF_SPLITS * RSKF_REPEATS,
    see config.py for constant definitions.

    Function relies on robust_train() function which can be used
    with Ridge and Lasso Cox-regression, can be written to function
    with Elastic Net as well.

    """

    subsampler: Subsampler = Subsampler.repeated_kfold(
        n_splits=CONFIG["tuning"]["rskf_splits"],
        n_repeats=CONFIG["tuning"]["rskf_repeats"],
    )
    sksurv_data: SksurvData = SksurvData(data_collection=data_collection)
    feature_no: int = sksurv_data.X.shape[1]
    try:
        art_X: np.ndarray[
            tuple[Any, ...], np.dtype[Any]
        ] = ArtificialGenerator(
            feature_no, ArtificialType.KNOCKOFF
        ).fit_transform(
            sksurv_data.X
        )
    except ArpackNoConvergence:
        return stability_selection(data_collection, plot, stop_requested)

    feature_names: list[str] = list(sksurv_data.X.columns)
    artificial_names: list[str] = [
        f"artificial_{i}" for i in range(feature_no)
    ]
    logger.info(
        "starting stability selectiont trial with"
        f"\nLOG_LAMBDA_MIN: {CONFIG['tuning']['log_lambda_min']}"
        f"\nLOG_LAMBDA_MAX: {CONFIG['tuning']['log_lambda_max']}"
        f"\nN_LAMBDA: {CONFIG['tuning']['n_lambda']}"
        f"\nN_CV_FOLDS: {CONFIG['tuning']['rskf_splits']}"
        f"\nN_REPEATS_IN_CV: {CONFIG['tuning']['rskf_repeats']}"
    )
    params: np.ndarray = np.logspace(
        CONFIG["tuning"]["log_lambda_min"],
        CONFIG["tuning"]["log_lambda_max"],
        CONFIG["tuning"]["n_lambda"],
    )
    results: Result = Result(
        feature_names + artificial_names,
        rounding_cutoff=int(CONFIG["tuning"]["coef_zero_cutoff"]),
        bin_min=int(CONFIG["tuning"]["log_lambda_min"]),
        bin_max=int(CONFIG["tuning"]["log_lambda_max"]),
    )
    for test, train in tqdm.tqdm(
        subsampler.split(art_X, sksurv_data.y),
        desc=f"{CONFIG['tuning']['rskf_splits']}-fold cross validation"
        + f"splits with {CONFIG['tuning']['rskf_repeats']} repeats",
        leave=False,
        total=CONFIG["tuning"]["rskf_splits"]
        * CONFIG["tuning"]["rskf_repeats"],
    ):
        if stop_requested and stop_requested():
            raise StabilitySelectionCancelled("Stopped by user.")
        for param in tqdm.tqdm(
            params,
            leave=False,
            desc=f"tuning on params 10^{CONFIG['tuning']['log_lambda_min']} "
            + f"to 10^{CONFIG['tuning']['log_lambda_max']} per fold",
        ):
            model: (
                lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float
            ) = robust_train(
                CONFIG["model"]["model_type"],
                art_X,
                sksurv_data.y,
                param,
                train,
                n_iter=CONFIG["tuning"]["n_trials"],
            )

            if isinstance(model, float):
                pass
            else:
                score: float | Any = model.score(
                    art_X[test, :], y=sksurv_data.y[test]
                )
                results.save_results(score, param, model.coef_.flatten())

        results.save_results_to_file()
    result: SingleStabilityResult = SingleStabilityResult(
        results.get_result_path()
    )
    if plot:
        result.plot_stability_path(save=True)
        result.plot_stability_path_with_thresh(save=True)
    return result

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sksurv.linear_model.coxph import CoxPHSurvivalAnalysis

from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.data_processing.result import Result

logger: logging.Logger = logging.getLogger(__name__)


def test_tiny_integration(
    model: CoxPHSurvivalAnalysis,
    tiny_true_data: tuple[
        pd.DataFrame, np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ],
) -> None:
    X, y = tiny_true_data
    feature_no: int = len(y)
    art_gen: ArtificialGenerator = ArtificialGenerator(
        len(y), ArtificialType.KNOCKOFF
    )
    model.fit(X, y)

    feature_names: list[str] = list(X.columns)
    artificial_names: list[str] = [
        f"artificial_{i}" for i in range(feature_no)
    ]
    art_X: np.ndarray[tuple[Any, ...], np.dtype[Any]] = art_gen.fit_transform(
        X
    )

    model.fit(art_X, y)
    results: Result = Result(feature_names + artificial_names)

    results.save_results(0.5, model.alpha, model.coef_)
    results.save_results(0.5, model.alpha, model.coef_)
    results.save_results(0.5, model.alpha, model.coef_)

    try:
        logger.info(results.get_long_result_df())
    except AssertionError:
        pass

    results.save_results(0.5, 0.1, model.coef_)
    logger.info(results.get_long_result_df())
    logger.info(results.get_results())


@pytest.mark.slow
def test_artificial_integration(
    model: CoxPHSurvivalAnalysis,
    full_true_data: tuple[
        pd.DataFrame, np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ],
) -> None:
    X, y = full_true_data
    feature_no: int = len(y)
    art_gen: ArtificialGenerator = ArtificialGenerator(
        len(y), ArtificialType.KNOCKOFF
    )
    model.fit(X, y)

    feature_names: list[str] = list(X.columns)
    artificial_names: list[str] = [
        f"artificial_{i}" for i in range(feature_no)
    ]
    art_X: np.ndarray[tuple[Any, ...], np.dtype[Any]] = art_gen.fit_transform(
        X
    )

    model.fit(art_X, y)

    results: Result = Result(feature_names + artificial_names)
    results.save_results(0.5, model.alpha, model.coef_)

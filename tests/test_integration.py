import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sksurv.linear_model.coxph import CoxPHSurvivalAnalysis

from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType
from survana.data_processing.result_models import Result

logger: logging.Logger = logging.getLogger(__name__)


def test__tiny_integration(
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
    top_features = results.get_top_features_and_save_frequencies(model.coef_)

    logger.info("-----------TOP FEATURES----------")
    info_str: str = ""
    for feature in top_features.keys():
        info_str += (
            f"\n\n{feature}\ncoef: {top_features[feature]}\nfreq:"
            + f"{results.feature_frequencies[feature]}"
        )

    logger.info(info_str)

    for feature in results.feature_frequencies.keys():

        assert (
            results.feature_frequencies[feature][0] == 1
            or results.feature_frequencies[feature][0] == 0
        ), "Expected binary value in feature frequency overview, "
        f"instead got {results.feature_frequencies[feature][0]}"

        assert (
            len(results.feature_frequencies[feature]) == 1
        ), "Expected all features to contain one element, "
        f"instead got {len(results.feature_frequencies[feature])}"


@pytest.mark.slow
def test__full_integration(
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
    results.get_top_features_and_save_frequencies(model.coef_)

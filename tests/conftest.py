from typing import Any

import numpy as np
import pandas as pd
import pytest
import sksurv.linear_model as lm

from config import PREFILTERED_DATA_PATH
from survana.data_processing.dataloaders import load_data_for_sksurv_coxnet


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.read_csv("test_data/test_matrix.csv", index_col="tests")


@pytest.fixture
def test_data_with_y() -> (
    tuple[pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[np.float64]]]
):
    data: pd.DataFrame = pd.read_csv(
        "test_data/test_matrix_with_y.csv", index_col="tests"
    )
    X: pd.DataFrame = data.iloc[:, 0:-2]
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]] = data.iloc[
        :, -2:
    ].to_records(
        index=False,
        column_dtypes={
            "d": bool,
            "e": "<f8",
        },
    )
    return X, y


@pytest.fixture
def test_result_data() -> tuple[
    list[int],
    list[str],
    list[str],
    np.ndarray[tuple[Any, ...], np.dtype[Any]],
]:
    true_features_str: list[str] = ["1", "2", "3"]
    true_features_int: list[int] = [1, 2, 3]
    fake_features_str: list[str] = ["fake_1", "fake_2", "fake_3"]

    coefs: np.ndarray = np.array([0, 0, 2, 3, 0, 1])
    return true_features_int, true_features_str, fake_features_str, coefs


@pytest.fixture
def model() -> lm.CoxPHSurvivalAnalysis:
    model = lm.CoxPHSurvivalAnalysis(
        verbose=True,
        alpha=0.2,
        n_iter=100,
    )
    return model


@pytest.fixture
def tiny_true_data() -> (
    tuple[pd.DataFrame, np.ndarray[tuple[Any, ...], np.dtype[Any]]]
):
    _, X, y = load_data_for_sksurv_coxnet(str(PREFILTERED_DATA_PATH))
    return pd.DataFrame(X.iloc[0:10, :10]), y[0:10]


@pytest.fixture
def full_true_data() -> (
    tuple[pd.DataFrame, np.ndarray[tuple[Any, ...], np.dtype[Any]]]
):
    _, X, y = load_data_for_sksurv_coxnet(str(PREFILTERED_DATA_PATH))
    return X, y

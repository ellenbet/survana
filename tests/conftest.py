from typing import Any

import numpy as np
import pandas as pd
import pytest


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

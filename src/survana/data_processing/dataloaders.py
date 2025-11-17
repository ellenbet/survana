# functions for loading our data
from typing import Any

import numpy as np
import pandas as pd


def load_data_for_sksurv_coxnet(
    path: str,
    separator: str = "\t",
    response_variables: tuple[str, str] = ("RFS_STATUS", "RFS_MONTHS"),
) -> tuple[
    pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
]:
    """Function to load data directly from preprocessed file
    (see preprocessing function), into sksurv coxnet package, which takes in a
    very specific type of array.

    Args:
        path (str): path to preprocessed data
        separator (str, optional):
            separator in preprocessed data. Defaults to "\t".
        response_variable (tuple[str, str]):
            bool + continious metric to measure
            outcome/response. Defaults to "RFS_STATUS", "RFS_MONTHS".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...],
        np.dtype[Any]]]: data plus design matrix and response
    """
    data: pd.DataFrame = pd.read_csv(path, sep=separator)
    X: pd.DataFrame = data.iloc[:, 2:-2]
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]] = data.iloc[
        :, -2:
    ].to_records(
        index=False,
        column_dtypes={
            response_variables[0]: bool,
            response_variables[1]: "<f8",
        },
    )
    return data, X, y

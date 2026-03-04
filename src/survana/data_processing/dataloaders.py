# functions for loading our data
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survana.config import CONFIG, PATHS

PREFILTERED_DATA_PATH: Path = PATHS["PREFILTERED_DATA_PATH"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def load_data_for_sksurv_coxnet(
    path: str,
    separator: str = "\t",
    response_variables: tuple[str, str] = (
        CONFIG["columns"]["censor_status"],
        CONFIG["columns"]["months_before_event"],
    ),
    exclude_columns: list[str] = [],
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
    logging.info(f"Found data from path {path}")

    return generate_compatable_data_collection(
        data,
        exclude_columns=exclude_columns,
        response_variables=response_variables,
    )


def load_partial_data_for_sksurv_coxnet(
    selected_features: list[str],
    path: str = str(PREFILTERED_DATA_PATH),
    separator: str = "\t",
    response_variables: tuple[str, str] = (
        CONFIG["columns"]["censor_status"],
        CONFIG["columns"]["months_before_event"],
    ),
    exclude_columns: list[str] = [],
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
    data: pd.DataFrame = pd.read_csv(
        path,
        sep=separator,
        usecols=selected_features
        + [response_variables[0], response_variables[1]],
    )
    logging.info(f"Found data from path {path}")
    return generate_compatable_data_collection(
        data,
        exclude_columns=exclude_columns,
        response_variables=response_variables,
    )


def generate_compatable_data_collection(
    data: pd.DataFrame,
    response_variables: tuple[str, str] = (
        CONFIG["columns"]["censor_status"],
        CONFIG["columns"]["months_before_event"],
    ),
    exclude_columns: list[str] = [],
) -> tuple[
    pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
]:
    """Function to load data directly, into sksurv coxnet package,
    which takes in a very specific type of array.

    Args:
        data (pd.DataFrame): loaded data with response variable inside df
        response_variable (tuple[str, str]):
            bool + continious metric to measure
            outcome/response. Defaults to "RFS_STATUS", "RFS_MONTHS".
        exclude_columns: list with cols to exclude

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...],
        np.dtype[Any]]]: data plus design matrix and response
    """

    for removed in exclude_columns:
        data.pop(removed)

    X: pd.DataFrame = data.drop(list(response_variables), axis=1)
    response: pd.DataFrame = data[list(response_variables)]
    y: np.recarray[
        tuple[Any, ...], np.dtype[np.float64]
    ] = response.to_records(
        index=False,
        column_dtypes={
            response_variables[0]: bool,
            response_variables[1]: "<f8",
        },
    )
    return data, X, y

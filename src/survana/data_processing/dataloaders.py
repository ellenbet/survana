# functions for loading our data
import logging
from typing import Any

import numpy as np
import pandas as pd

from survana.config import CONFIG, PATHS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def load_data_for_sksurv_coxnet(
    path_to_epigenetic_features: str = str(PATHS["PREFILTERED_DATA_PATH"]),
    path_to_clinical_data: str = str(PATHS["CLINICAL_DATA_PATH"]),
    separator: str = CONFIG["columns"]["sep"],
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

    logger.info(
        f"Loading data from paths:\n{path_to_epigenetic_features}"
        + f"\n{path_to_clinical_data}"
    )
    logger.info(f"Using response variables from columns: {response_variables}")
    epigen_data = pd.read_csv(path_to_epigenetic_features, index_col=0)

    assert not epigen_data.isna().any(axis=None), (
        "Feature dataframe contains NaN values, impute missing values with"
        + " survana.data_pre_filtering.knn_imputer to fix."
    )

    clinical_data: pd.DataFrame = pd.read_csv(
        path_to_clinical_data,
        sep=separator,
        index_col="PATIENT_ID",
        usecols=["PATIENT_ID"] + list(response_variables),
    )

    assert not clinical_data.isna().any(axis=None), (
        "Clinical dataframe contains NaN values, impute missing values with"
        + " survana.data_pre_filtering.knn_imputer to fix."
    )

    data = pd.concat([epigen_data, clinical_data], axis=1, join="inner")
    logger.info(
        f"Starting selection with final design matrix shape {data.shape}"
    )

    return generate_compatable_data_collection(
        data,
        exclude_columns=exclude_columns,
        response_variables=response_variables,
    )


def load_partial_data_for_sksurv_coxnet(
    selected_features: list[str],
    path_to_epigenetic_features: str = str(PATHS["PREFILTERED_DATA_PATH"]),
    path_to_clinical_data: str = str(PATHS["CLINICAL_DATA_PATH"]),
    separator: str = CONFIG["columns"]["sep"],
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

    selected_true = [
        feature for feature in selected_features if "artificial" not in feature
    ]

    epigen_data: pd.DataFrame = pd.read_csv(
        path_to_epigenetic_features,
        index_col="PATIENT_ID",
        usecols=["PATIENT_ID"] + selected_true,
    )

    epigen_data.set_index(epigen_data.columns[0])

    clinical_data: pd.DataFrame = pd.read_csv(
        path_to_clinical_data,
        sep=separator,
        index_col="PATIENT_ID",
        usecols=["PATIENT_ID", response_variables[0], response_variables[1]],
    )

    data = pd.concat([epigen_data, clinical_data], axis=1, join="inner")
    assert len(data) > 0, "DataFrame is empty, inner join has failed."

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

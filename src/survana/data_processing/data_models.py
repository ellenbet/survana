import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field

from config import CENSOR_STATUS

logger: logging.Logger = logging.getLogger(__name__)


class SksurvData(BaseModel):
    """Container for survival-analysis data and related utilities.

    This class stores three main inputs:
      - a pandas DataFrame containing raw data,
      - a pandas DataFrame containing the design matrix (features),
      - a NumPy structured recarray containing the survival response
        (event indicator + time).

    Upon initialization, the model computes several derived attributes,
    including `data`, `X`, `y`, `censored_patients_percentage`,
    `y_censored`, and `y_survival`.

    The class also provides convenience methods for generating
    stratified cross-validation split iterators, including standard
    `StratifiedKFold` and `RepeatedStratifiedKFold`. Additionally,
    it provides a utility to extract the most influential features
    from model coefficients.

    Attributes:
        data (pd.DataFrame):
            Raw dataset used for analysis.
        X (pd.DataFrame):
            Design matrix of predictor variables.
        y (np.recarray):
            Structured survival response array
            (event indicator and survival time).
        y_censored (np.ndarray):
            Boolean array indicating censoring status
            (0 = event occurred, 1 = censored).
        y_survival (np.ndarray):
            Array of survival times.
        censored_patients_percentage (float):
            Proportion of samples that are censored.

    Methods:
        stratified_kfold_splits(X, y, n_splits, random_state, shuffle):
            Creates a `StratifiedKFold` iterator using either instance data
            or the provided `X` and `y`.
        stratified_repeated_kfold_splits(X, y, n_repeat, n_splits, random_s):
            Creates a `RepeatedStratifiedKFold` iterator using either
            instance data or the provided `X` and `y`.
        get_best_features(all_coefs):
            Returns a dictionary containing the highest-magnitude model
            coefficients and their corresponding feature names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ]

    @computed_field  # type: ignore[misc]
    @property
    def data(self) -> pd.DataFrame:
        return self.data_collection[0]

    @computed_field  # type: ignore[misc]
    @property
    def X(self) -> pd.DataFrame:
        return self.data_collection[1]

    @computed_field  # type: ignore[misc]
    @property
    def y(self) -> np.recarray[tuple[Any, ...], np.dtype[np.float64]]:
        return self.data_collection[2]

    @computed_field  # type: ignore[misc]
    @property
    def censored_patient_percentage(self) -> float:
        return (
            len(self.data[CENSOR_STATUS]) - self.data[CENSOR_STATUS].sum()
        ) / len(self.data[CENSOR_STATUS])

    @computed_field  # type: ignore[misc]
    @property
    def y_censored(self) -> np.ndarray:
        return np.array([datapoint[0] for datapoint in self.y])  # type: ignore

    @computed_field  # type: ignore[misc]
    @property
    def y_survival(self) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        return np.array([datapoint[1] for datapoint in self.y])

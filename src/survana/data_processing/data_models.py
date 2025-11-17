import logging
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config import CENSOR_STATUS, COEF_CUTOFF

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
        stratified_kfold_splits(X=None, y=None, n_splits=5, random_state=None,
        shuffle=False):
            Creates a `StratifiedKFold` iterator using either instance data
            or the provided `X` and `y`.
        stratified_repeated_kfold_splits(X=None, y=None, n_repeats=2,
        n_splits=5, random_state=None):
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

    def stratified_kfold_splits(
        self,
        X: pd.DataFrame | None = None,
        y: np.recarray[tuple[Any, ...], np.dtype[np.float64]] | None = None,
        n_splits: int = 5,
        random_state=None,
        shuffle: bool = False,
    ) -> Iterator[Any]:  # type: ignore
        """Makes a stratified k fold splitting iterator based on
        either instance variables or argument X and y

        Args:
            X (pd.DataFrame | pd.Series | None, optional): design matrix.
            Defaults to None.
            y (np.recarray[tuple[Any, ...], np.dtype[np.float64]]
            | None, optional): response variable. Defaults to None.
            n_splits (int, optional): amount of split for cv. Defaults to 5.
            random_state (_type_, optional): random state. Defaults to None.
            shuffle (bool): Defaults to False.

        Returns:
            Iterator: iterator with fold, (train, test) indexes.

        Yields:
            Iterator[Any]: _description_
        """
        skf: StratifiedKFold = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle
        )
        if X is None and y is None:
            return skf.split(self.X, self.y_censored)
        else:
            return skf.split(X, y)  # type: ignore

    def stratified_repeated_kfold_splits(
        self,
        X: pd.DataFrame | pd.Series | None = None,
        y: np.recarray[tuple[Any, ...], np.dtype[np.float64]] | None = None,
        n_repeats: int = 2,
        n_splits: int = 5,
        random_state=None,
    ) -> Iterator[Any]:  # type: ignore
        """Makes a stratified repeated k fold splitting iterator based on
        either instance variables or argument X and y

        Args:
            X (pd.DataFrame | pd.Series | None, optional): design matrix.
            Defaults to None.
            y (np.recarray[tuple[Any, ...], np.dtype[np.float64]]
            | None, optional): response variable. Defaults to None.
            n_repeats (int, optional): amount of repeats for repeated kfold.
            Defaults to 2.
            n_splits (int, optional): amount of split for cv. Defaults to 5.
            random_state (_type_, optional): random state. Defaults to None.

        Returns:
            Iterator: iterator with fold, (train, test) indexes.

        Yields:
            Iterator[Any]: _description_
        """
        rskf: RepeatedStratifiedKFold = RepeatedStratifiedKFold(
            n_repeats=n_repeats,
            n_splits=n_splits,
            random_state=random_state,
        )
        if X is None and y is None:
            return rskf.split(self.X, self.y_censored)
        else:
            y_cens: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
                [datapoint[0] for datapoint in y]  # type: ignore
            )
            try:
                assert int(X.shape[0]) == int(y_cens.shape[0])  # type: ignore
            except AssertionError:
                logger.error(
                    "Design matrix shape X with shape"
                    + f" {X.shape[0]}"  # type: ignore
                    + " does not match shape of response variable"
                    + f" y with shape {y_cens.shape[0]}"
                )
            return rskf.split(X, y_cens)  # type: ignore

    # TODO add MC CV method

    def get_best_features(
        self, all_coefs: np.ndarray
    ) -> dict[str, np.float64]:
        """Method that takes inn coefs found from model, sorts on largest
        absolute value and returns a dictionary with keys as features
        and values as coefficients

        Args:
            all_coefs (np.ndarray): from model training, expects model.coefs_
            from Sksurv package

        Returns:
            dict[str, np.float64]: keys as feature, value as coef
        """
        indexed_coefs: list[tuple[int, np.float64]] = list(
            enumerate(all_coefs)
        )
        sorted_coefs_with_index: list[tuple[int, np.float64]] = sorted(
            indexed_coefs,
            key=lambda x: abs(x[1]),  # type:ignore
            reverse=True,
        )[:COEF_CUTOFF]
        top_features: dict[str, np.float64] = {}
        for ind, coef_val in sorted_coefs_with_index:
            top_features[str(self.X.columns[ind])] = round(coef_val, 5)

        return top_features

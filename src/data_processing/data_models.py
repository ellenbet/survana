import logging
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config import CENSOR_STATUS

logger: logging.Logger = logging.getLogger(__name__)


class SksurvData(BaseModel):
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
        skf: StratifiedKFold = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle
        )
        if X is None and y is None:
            return skf.split(self.X, self.y_censored)
        else:
            return skf.split(X, y)  # type: ignore

    def stratified_repeated_kfold_splits(
        self,
        X: pd.DataFrame | None = None,
        y: np.recarray[tuple[Any, ...], np.dtype[np.float64]] | None = None,
        n_repeats: int = 2,
        n_splits: int = 5,
        random_state=None,
    ) -> Iterator[Any]:  # type: ignore
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

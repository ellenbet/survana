from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

logger: logging.Logger = logging.getLogger(__name__)


class SubsamplingStrategy(StrEnum):
    """Subsampling stratergies allowed in the Subsampler class,
    this is an Sklearn _BaseKfold wrapper for three different
    stratified splitters.

    Attrinbutes:
        KFOLD : "kfold"
        REPEATED_KFOLD : "repeated_kfold"
        SHUFFLE_SPLIT : "shuffle_split"
    """

    KFOLD = "kfold"
    REPEATED_KFOLD = "repeated_kfold"
    SHUFFLE_SPLIT = "shuffle_split"


class Subsampler:
    def __init__(
        self,
        subsampling: SubsamplingStrategy,
        n_splits: int = 5,
        n_repeats: int = 0,
        test_size: float = 0,
        shuffle: bool = False,
        random_state: None | int = None,
    ) -> None:
        self.subsampling: SubsamplingStrategy = subsampling
        self.n_splits: int = n_splits
        self.n_repeats: int = n_repeats
        self.test_size: float = test_size
        self.shuffle: bool = shuffle
        self.random_state: None | int = random_state

    @classmethod
    def kfold(
        cls,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: None | int = None,
    ) -> Subsampler:
        return cls(
            subsampling=SubsamplingStrategy.KFOLD,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    @classmethod
    def repeated_kfold(
        cls,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: None | int = None,
    ) -> Subsampler:
        return cls(
            subsampling=SubsamplingStrategy.REPEATED_KFOLD,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    @classmethod
    def shuffle_split(
        cls,
        test_size: float = 0.2,
        n_splits: int = 1,
        random_state: None | int = None,
    ) -> Subsampler:
        return cls(
            subsampling=SubsamplingStrategy.SHUFFLE_SPLIT,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
        )

    def split(self, X, y) -> Iterator[Any]:
        if self.subsampling == SubsamplingStrategy.REPEATED_KFOLD:
            return stratified_repeated_kfold_splits(
                X, y, self.n_repeats, self.n_splits, self.random_state
            )
        elif self.subsampling == SubsamplingStrategy.SHUFFLE_SPLIT:
            return stratified_shuffle_split(
                X, y, self.n_splits, self.test_size, self.random_state
            )
        else:
            return stratified_kfold_splits(
                X, y, self.n_splits, self.random_state, self.shuffle
            )


def stratified_kfold_splits(
    X: pd.DataFrame,
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]],
    n_splits: int = 5,
    random_state=None,
    shuffle: bool = False,
) -> Iterator[Any]:  # type: ignore
    """Makes a stratified k fold splitting iterator for X and y

    Args:
        X (pd.DataFrame | pd.Series):
            design matrix..
        y (np.recarray[tuple[Any, ...], np.dtype[np.float64]]):
            response variable.
        n_splits (int, optional): amount of split for cv. Defaults to 5.
        random_state (_type_, optional): random state. Defaults to None.
        shuffle (bool): Defaults to False.

    Returns:
        Iterator: iterator with fold, (train, test) indexes.

    Yields:
        Iterator[Any]: Containing split indexes
    """
    skf: StratifiedKFold = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle
    )
    y_cens: np.ndarray[tuple[Any, ...], np.dtype[Any]] = _get_y_cens(
        X.shape, y
    )
    return skf.split(X, y_cens)  # type: ignore


def stratified_repeated_kfold_splits(
    X: pd.DataFrame | pd.Series,
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]],
    n_repeats: int = 2,
    n_splits: int = 5,
    random_state=None,
) -> Iterator[Any]:  # type: ignore
    """Makes a stratified repeated k fold splitting iterator for X and y

    Args:
        X (pd.DataFrame | pd.Series): design matrix.
        y (np.recarray[tuple[Any, ...], np.dtype[np.float64]]):
            response variable.
        n_repeats (int, optional):
            amount of repeats for repeated kfold. Defaults to 2.
        n_splits (int, optional): amount of split for cv. Defaults to 5.
        random_state (_type_, optional): random state. Defaults to None.

    Returns:
        Iterator: iterator with fold, (train, test) indexes.

    Yields:
        Iterator[Any]: Containing split indexes
    """
    rskf: RepeatedStratifiedKFold = RepeatedStratifiedKFold(
        n_repeats=n_repeats,
        n_splits=n_splits,
        random_state=random_state,
    )

    y_cens: np.ndarray[tuple[Any, ...], np.dtype[Any]] = _get_y_cens(
        X.shape, y
    )
    return rskf.split(X, y_cens)  # type: ignore


def stratified_shuffle_split(
    X: pd.DataFrame,
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]],
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: None | int = None,
) -> Iterator[Any]:  # type: ignore
    """Stratified shuffle split can also be described as stratified Monte
    Carlo cross validation, creates random test and train partitions that
    are potentially overlapping.

    Args:
        X (pd.DataFrame):
            Design matrix.
        y (np.recarray[tuple[Any, ...], np.dtype[np.float64]]):
            Response variable.
        n_splits (int, optional):
            Number of splits/subsamples. Defaults to 5.
        test_size (float, optional):
            Test percentage. Defaults to 0.2.
        train_size (float, optional):
            Train percentage. Defaults to None, takes on shape 1-test_size.
        random_state (_type_, optional):
            Random state. Defaults to None.

    Returns:
        Iterator: To be looped through in hyperparameter tuning.

    Yields:
        Iterator[Any]: Containing split indexes
    """
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    y_cens: np.ndarray[tuple[Any, ...], np.dtype[Any]] = _get_y_cens(
        X.shape, y
    )
    return sss.split(X, y_cens)


def _get_y_cens(
    X_shape: tuple[int, int],
    y: np.recarray[tuple[Any, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
    """Gets y_cens for later stratified splits

    Args:
        X_shape (tuple[int, int]): shape of design matrix
        y (
        np.recarray[tuple[Any, ...], np.dtype[np.float64]]):
            response variable

    Returns:
        np.ndarray[tuple[Any, ...], np.dtype[Any]]:
            first column of respose variable
    """
    y_cens: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
        [datapoint[0] for datapoint in y]  # type: ignore
    )

    assert int(X_shape[0]) == int(y_cens.shape[0]), (
        "Design matrix X with rows:"
        + f" {X_shape[0]}"  # type: ignore
        + " does not match response variable"
        + f" y rows: {y_cens.shape[0]}"
    )  # type: ignore

    return y_cens

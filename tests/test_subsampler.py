from typing import Any, Iterator

import numpy as np
import pandas as pd

from survana.data_processing.data_subsampler import Subsampler


def test_kfold(
    test_data_with_y: tuple[pd.DataFrame, np.recarray],
) -> None:
    X, y = test_data_with_y
    sub: Subsampler = Subsampler.kfold(n_splits=2)
    splits = sub.split(X, y)

    assert X.shape == (4, 3), f"Expected X shape (4, 3), got {X.shape}"
    assert y.shape == (4,), f"Expected y shape (4, 2), got {y.shape}"

    counter = 0
    for i, (inner_train_ind, inner_test_ind) in enumerate(splits):
        assert len(inner_test_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of test ind to be half of row count"
        assert len(inner_train_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of traan ind to be half of row count"
        counter += 1

    assert counter == 2, "Expected counter to match number of splits"


def test_repeated_kfold(
    test_data_with_y: tuple[pd.DataFrame, np.recarray],
) -> None:
    X, y = test_data_with_y
    sub: Subsampler = Subsampler.repeated_kfold(n_splits=2, n_repeats=1)
    splits = sub.split(X, y)

    counter = 0
    for i, (inner_train_ind, inner_test_ind) in enumerate(splits):
        assert len(inner_test_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of test ind to be half of row count"
        assert len(inner_train_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of traan ind to be half of row count"
        counter += 1

    assert counter == 2, "Expected counter to match number of splits"


def test_shuffle_splits(
    test_data_with_y: tuple[pd.DataFrame, np.recarray],
) -> None:
    X, y = test_data_with_y
    sub: Subsampler = Subsampler.shuffle_split(n_splits=2, test_size=0.5)
    splits = sub.split(X, y)

    counter = 0
    for i, (inner_train_ind, inner_test_ind) in enumerate(splits):
        assert len(inner_test_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of test ind to be half of row count"
        assert len(inner_train_ind) == int(
            X.shape[0] / 2
        ), "Expected amount of traan ind to be half of row count"
        counter += 1

    assert counter == 2, "Expected counter to match number of splits"


def test_y_cens_dimension_check(
    test_data_with_y: tuple[pd.DataFrame, np.recarray],
) -> Iterator[Any] | None:
    X, y = test_data_with_y
    sub: Subsampler = Subsampler.repeated_kfold(n_splits=2, n_repeats=1)
    try:
        splits = sub.split(X.iloc[1:, :], y)
    except AssertionError:
        return None

    assert (
        1 == 2
    ), "Splitting on two differnt row dimension was not caught by Subsampler"
    return splits

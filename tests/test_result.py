from typing import Any

import numpy as np

from survana.data_processing.result_models import Result


def test_result_with_int(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
    ],
) -> None:
    true_int, _, fake_str, coefs = test_result_data

    result: Result = Result(true_int + fake_str)
    top_features: dict[
        str, float
    ] = result.get_top_features_and_save_frequencies(coefs)

    assert len(top_features.keys()) == 3, (
        f"Expected top features to have 3 elements, "
        f"got {len(top_features.keys())}"
    )

    for feature_name, true_freq in zip(
        result.feature_frequencies.keys(), [0, 0, 1, 1, 0, 1]
    ):
        assert len(result.feature_frequencies[feature_name]) == 1, (
            f"Expected length of 1 for all frequency counter lists, "
            f"got {len(result.feature_frequencies[feature_name])} instead."
        )
        freq = result.feature_frequencies[feature_name][0]
        assert freq == true_freq, (
            f"Found frequency/binary indicator not valid, got {freq} "
            f"but expected {true_freq}"
        )

    result.save_frequencies(coefs)

    assert len(top_features.keys()) == 3, (
        f"Expected top features to have 3 elements, "
        f"got {len(top_features.keys())}"
    )

    for feature_name, true_freq in zip(
        result.feature_frequencies.keys(), [0, 0, 1, 1, 0, 1]
    ):
        assert len(result.feature_frequencies[feature_name]) == 2, (
            f"Expected length of 1 for all frequency counter lists, "
            f"got {len(result.feature_frequencies[feature_name])} instead."
        )
        frequency: np.floating[Any] = np.mean(
            result.feature_frequencies[feature_name]
        )
        assert frequency == true_freq, (
            f"Found frequency/binary indicator not valid, got {frequency} "
            f"but expected {true_freq}"
        )


def test_result_assert(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
    ],
) -> None:
    true_int, true_str, fake_str, coefs = test_result_data
    try:
        result: Result = Result(true_int)
        result.get_top_features_and_save_frequencies(coefs)
    except AssertionError:
        return None

    assert 1 == 2, "Assert check not passed in Result class"

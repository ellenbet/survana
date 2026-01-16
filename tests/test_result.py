from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from survana.data_processing.result_models import Result


def test_result_dict(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    true_int, true_int_str, fake_str, coefs, hyperparam = test_result_data

    result: Result = Result(true_int_str + fake_str)
    for hp in hyperparam:
        assert (
            result.results[result.feature_names[0], hp]["count"] == 0
        ), "count not 0"
        assert (
            result.results[result.feature_names[0], hp]["occurence"] == 0
        ), "occurence not 0"

    result.save_results(hyperparam[0], coefs)

    for hp in hyperparam[1:]:
        assert (
            result.results[result.feature_names[0], hp]["count"] == 0
        ), "count not 0"
        assert (
            result.results[result.feature_names[0], hp]["occurence"] == 0
        ), "occurence not 0"

    for feature_name, true_freq in zip(
        result.feature_names, [1, 1, 1, 0, 0, 0]
    ):
        o: int = result.results[feature_name, result.get_bin(hyperparam[0])][
            "occurence"
        ]
        c: int = result.results[feature_name, result.get_bin(hyperparam[0])][
            "count"
        ]
        assert o == true_freq, (
            f"Expected occurence to be {true_freq}"
            f" for {feature_name}, got"
            f" {o} "
            "instead."
        )

        assert c == 1, (
            f"Expected count to be {true_freq} for {feature_name}, "
            f"got {c}."
        )


def test_occurence_increase(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    _, true_int_str, fake_str, coefs, hyperparams = test_result_data
    result: Result = Result(true_int_str + fake_str)
    hyperparam: float = result.get_bin(hyperparams[0])
    result.save_results(hyperparam, coefs)
    result.save_results(hyperparam, coefs)
    o_1 = result.results[(result.feature_names[0], hyperparam)]["occurence"]
    assert o_1 == 2, "true 1 occurence failed, got " + f" {o_1}"
    assert (
        result.results[(true_int_str[0], hyperparam)]["count"] == 2
    ), "true 1 count failed"
    assert (
        result.results[("fake_1", hyperparam)]["occurence"] == 0
    ), "fake 1 occurence failed"
    assert (
        result.results[("fake_1", hyperparam)]["count"] == 2
    ), "fake 1 count failed"


def test_save_and_get_results(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    result: Result = Result(true_str)
    result.save_results(hyperparam[0], coefs)
    result.get_results()


def test_result_names(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    try:
        Result(true_str + [1])
    except AssertionError:
        return
    pytest.fail("Non-str name allowed in result")


def test_plot_results(
    monkeypatch,
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    result: Result = Result(true_str)
    result.save_results(hyperparam[0], coefs)
    result.save_results(hyperparam[1], coefs)
    result.save_results(hyperparam[2], coefs)
    result.plot_results()


def test_failure_plot_results(
    monkeypatch,
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    result: Result = Result(true_str)
    try:
        result.plot_results()
    except AssertionError:
        return None

    pytest.fail("Failed to catch empty result plotting")

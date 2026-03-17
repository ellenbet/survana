import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from survana.config import PATHS
from survana.data_processing.data_models import SksurvData
from survana.data_processing.dataloaders import (
    load_partial_data_for_sksurv_coxnet,
)
from survana.result_processing.multiple_stability_result import (
    MultipleStabilityResult,
)
from survana.result_processing.result import Result
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)
from survana.tuning.post_stability_selection import coxph_final_tuning_optuna

warnings.filterwarnings("ignore", category=UserWarning)

logger: logging.Logger = logging.getLogger(__name__)


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

    result.save_results(0.5, hyperparam[0], coefs)

    for hp in hyperparam[1:]:
        assert (
            result.results[result.feature_names[0], hp]["count"] == 0
        ), "count not 0"
        assert (
            result.results[result.feature_names[0], hp]["occurence"] == 0
        ), "occurence not 0"
        assert (
            result.results[result.feature_names[0], hp]["accumulated_score"]
            == 0
        ), "score not 0.5"

    for feature_name, true_freq in zip(
        result.feature_names, [1, 1, 1, 0, 0, 0]
    ):
        o: int | float = result.results[
            feature_name, result.get_bin(hyperparam[0])
        ]["occurence"]
        c: int | float = result.results[
            feature_name, result.get_bin(hyperparam[0])
        ]["count"]
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
    result.save_results(0.5, hyperparam, coefs)
    result.save_results(0.5, hyperparam, coefs)
    o_1: int | float = result.results[(result.feature_names[0], hyperparam)][
        "occurence"
    ]
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


def test_score(
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
    result.save_results(0, hyperparam, coefs)
    result.save_results(1, hyperparam, coefs)
    o_1: int | float = result.results[(result.feature_names[0], hyperparam)][
        "accumulated_score"
    ]
    assert o_1 == 1, "accumulated score failed, got " + f" {o_1}"

    long_results = result.get_long_result_df()
    mean_score = long_results["average_score"][0]
    assert mean_score == 0.5, f"expected mean score 0.5, got {mean_score}"


def test_save_and_get_results(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    _, true_str, _, coefs, hyperparam = test_result_data
    result: Result = Result(true_str)
    result.save_results(0.5, hyperparam[0], coefs)
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


def test_save_and_plot_results(
    monkeypatch: pytest.MonkeyPatch,
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    result: Result = Result(true_str, bin_min=-2, bin_max=1)
    result.save_results(0.5, hyperparam[0], coefs)
    result.save_results(0.5, hyperparam[1], coefs)
    result.save_results(0.5, hyperparam[2], coefs)


def test_hyperparam_checker(
    test_result_data: tuple[
        list[int],
        list[str],
        list[str],
        np.ndarray[tuple[Any, ...], np.dtype[Any]],
        list[float],
    ],
) -> None:
    true_int, true_str, fake_str, coefs, hyperparam = test_result_data
    result: Result = Result(true_str, bin_min=-1, bin_max=1)
    try:
        result.save_results(0.5, 10**-2, coefs)
    except AssertionError:
        pass

    try:
        result.save_results(0.5, 10**2, coefs)
    except AssertionError:
        return None

    pytest.fail("Failed to catch invalid hyperparam value")


def test_plotter_plot_results(
    monkeypatch,
    test_result: Path,
) -> None:
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    plotter = SingleStabilityResult(test_result)
    plotter.plot_stability_path()
    plotter.plot_top_exponent()
    plotter.plot_top_freq_dist()
    plotter.plot_min_fdr()
    plotter.plot_stability_path_with_thresh()


def test_post_stability_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MultipleStabilityResult,
        "_write_to_json_file",
        lambda *args, **kwargs: None,
    )
    accumulated_results = MultipleStabilityResult()
    single_result: SingleStabilityResult = SingleStabilityResult(
        PATHS["RESULT_CSV_DATA_PATH"] / "log(lambda)_-8_to_1_results__df.csv"
    )

    assert len(single_result.get_selected_features()) > 0, "no features found"
    new_data_collection: tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.recarray[tuple[Any, ...], np.dtype[Any]],
    ] = load_partial_data_for_sksurv_coxnet(
        single_result.get_selected_features()
    )

    final_run_tuning: dict[
        str, dict[str, Any] | np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ] = coxph_final_tuning_optuna(
        data=SksurvData(data_collection=new_data_collection)
    )

    accumulated_results.add_single_result(single_result)
    accumulated_results.add_model_score(
        final_run_tuning, features=single_result.get_selected_features()
    )

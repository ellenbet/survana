from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import COEF_ZERO_CUTOFF, LOG_LAMBDA_MAX, LOG_LAMBDA_MIN


class Result:
    """Result class to compute frequencies per resulting
    np.array of coefficients found post test and train.

    Attributes:
        feature_names (np.ndarray | list[Any]):
            collection of feature names, length needs to match
            number of coefficients.
        rounding_cutoff (int):
            to be used in round() function, determines whether
            feature coefficient value is zero or non-zero, which
            translates to feature being selected or not.
        bin_min (int): smallest 10-log hyperparameter, defaults to -5
        bin_max (int): highest 10-log hyperparameter, defaults to 1
    """

    def __init__(
        self,
        feature_names: np.ndarray | list[str],
        rounding_cutoff: int = COEF_ZERO_CUTOFF,
        coefficient_number_cutoff: int | None = None,
        bin_min: int = LOG_LAMBDA_MIN,
        bin_max: int = LOG_LAMBDA_MAX,
        bin_res: int = 10,
    ) -> None:
        assert all(
            isinstance(name, str) for name in feature_names
        ), "feature names must be str"

        self.results: dict[tuple[str, float], dict[str, int]] = defaultdict(
            lambda: {"occurence": 0, "count": 0}
        )
        self.feature_names = feature_names
        self.rounding_cutoff = rounding_cutoff
        self.coefficient_number_cutoff: int | None = coefficient_number_cutoff
        self.bin_max = bin_max
        self.bin_min = bin_min
        self._hyperparam_bin_configuration(bin_min, bin_max, bin_res)

    def save_results(self, hyperparam: float, all_coefs: np.ndarray) -> None:
        """Method that saves feature selection frequency by defining all
        non-zero features as selected. Non-zero cutoff is defined by
        rounding_cutoff attribute.

        Args:
            hyperparam (float): in this case lambda
            all_coefs (np.ndarray): coefs from regression
        """
        hyperparam_bin: float = self._get_bin(hyperparam)
        for coef, feature_name in zip(all_coefs, self.feature_names):
            self.results[(feature_name, hyperparam_bin)][
                "occurence"
            ] += self._above_cutoff(coef)
            self.results[(feature_name, hyperparam_bin)]["count"] += 1

    def get_results(self) -> dict[tuple[str, float], dict[str, int]]:
        """Result method asserts length of both result objects are
        equal before returning results.

        Returns:
            tuple[dict[str, list[int]], list[Any]]:
            feature frequencies and hyperparameters
        """
        return self.results

    def get_long_result_df(self) -> pd.DataFrame:
        assert len(self.results) > 1, "Results dict empty or 1"
        long_df = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "hyperparam": hyperparam_bin,
                    "occurence": stats["occurence"],
                    "count": stats["count"],
                }
                for (feature, hyperparam_bin), stats in self.results.items()
            ]
        )

        long_df["freq"] = long_df["occurence"] / long_df["count"]
        return long_df

    def get_results_file(self) -> None:
        long_df = self.get_long_result_df()
        long_df.to_csv("results_df.csv")

    def plot_results(self) -> None:
        """Method that plots stability paths"""
        assert len(self.results) > 1, "Results dict empty or 1"
        long_df = self.get_long_result_df()
        wide_df = long_df.pivot(
            index="hyperparam", columns="feature", values="freq"
        )
        wide_df = wide_df.copy()

        wide_df.index = 1.0 / wide_df.index.astype(float)
        wide_df = wide_df.sort_index()

        wide_df.plot(legend=False)
        plt.xscale("log")
        plt.xlabel(r"(1 / λ)-bins")
        plt.ylabel(r"feature selection frequency(θ)")
        plt.show()

    def _above_cutoff(self, x: float) -> int:
        """Private function to sett coefs to zero if below cutoff

        Args:
            x (float): coefficient

        Returns:
           occurence (int): 0 or 1
        """

        if abs(round(x, self.rounding_cutoff)) > 0:
            return 1
        else:
            return 0

    def _hyperparam_bin_configuration(
        self,
        min: int,
        max: int,
        res: int,
    ) -> None:
        """Private method that configures the bin logic for later
        stability path plotting of continious hyperparameters

        Args:
            min (int): minimum 10-log of hyperparam option
            max (int): maximum 10-log of hyperparam option
            res (int): resolution between each 10-log difference
        """

        self._resolution: int = (max - min) * res
        self._bins: np.ndarray[
            tuple[Any, ...], np.dtype[np.float64]
        ] = np.logspace(min, max, self._resolution)
        self._names = np.array(
            [
                (self._bins[i - 1] + self._bins[i]) * 0.5
                for i in range(1, len(self._bins))
            ]
        )

    def _get_bin(
        self,
        hyperparam: float,
    ) -> float:
        """Private methods that returns the bin belonging
        to the hyperparameter.

        Args:
            hyperparam (float): hyperparameter

        Returns:
            float: name of bin
        """
        log_hyp = np.log10(hyperparam)
        assert log_hyp >= self.bin_min and hyperparam <= self.bin_max, [
            f"non-valid log(hyperparameter) {log_hyp}, must be between "
            + f"{self.bin_min} and {self.bin_max}"
        ]
        return self._names[
            np.searchsorted(self._bins[:], [hyperparam], side="left") - 1
        ][0]

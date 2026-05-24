import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from survana.config import CONFIG, PATHS

COEF_ZERO_CUTOFF, LOG_LAMBDA_MAX, LOG_LAMBDA_MIN = (
    CONFIG["tuning"]["coef_zero_cutoff"],
    CONFIG["tuning"]["log_lambda_max"],
    CONFIG["tuning"]["log_lambda_min"],
)

LOG_LAMBDA_MIN = CONFIG["tuning"]["log_lambda_min"]
LOG_LAMBDA_MAX = CONFIG["tuning"]["log_lambda_max"]
N_LAMBDA = CONFIG["tuning"]["n_lambda"]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


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
        feature_names: list[str],
        rounding_cutoff: int = COEF_ZERO_CUTOFF,
    ) -> None:
        assert all(
            isinstance(name, str) for name in feature_names
        ), "feature names must be str"

        self.results: dict[
            tuple[str, float], dict[str, int | float]
        ] = defaultdict(lambda: {"occurence": 0, "count": 0})
        self.feature_names: list[str] = feature_names
        self.trial_name: str = (
            f"log(lambda)_{LOG_LAMBDA_MIN}" + f"_to_{LOG_LAMBDA_MAX}_results_"
        )
        self.rounding_cutoff: int = rounding_cutoff

    def save_results(self, hyperparam: float, all_coefs: np.ndarray) -> None:
        """Method that saves feature selection frequency by defining all
        non-zero features as selected. Non-zero cutoff is defined by
        rounding_cutoff attribute.

        Args:
            hyperparam (float): in this case lambda
            all_coefs (np.ndarray): coefs from regression
        """

        for coef, feature_name in zip(all_coefs, self.feature_names):
            self.results[(feature_name, hyperparam)][
                "occurence"
            ] += self._above_cutoff(coef)
            self.results[(feature_name, hyperparam)]["count"] += 1

    def get_results(self) -> dict[tuple[str, float], dict[str, int | float]]:
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

    def save_results_to_file(self) -> None:
        """Saves result dataframe as .csv in result csv path set in
        config.toml"""
        long_df = self.get_long_result_df()
        filename = f"{self.trial_name}_df.csv"
        self.result_path: Path = PATHS["BASE_DIR"] / (
            PATHS["RESULT_CSV_DATA_PATH"] / filename
        )
        long_df.to_csv(self.result_path)
        self.long_df: pd.DataFrame = long_df

    def get_result_path(self) -> Path:
        return self.result_path

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

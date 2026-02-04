import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from survana.config import PATHS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


class Plotter:
    """Class to plot and process
    all data from the result .csv generated from stability function.

    Methods:
        plot_stability_path(save:bool = False)
        plot_top_exponent(save:bool = False)
        plot_top_freq_dist(save:bool = False)
        plot_min_fdr(save:bool = False)
        plot_stability_path_with_thresh(save:bool = False)

    """

    def __init__(self, full_result_path: Path):
        _set_plt_params()
        p: Path = PATHS["RESULT_CSV_DATA_PATH"]
        self.fig_path: Path = PATHS["RESULT_FIGURES_DATA_PATH"]
        self.path: str = (
            str(full_result_path)
            .replace(
                f"{p}/",
                "",
            )
            .replace(".csv", "")
        )
        self.data: pd.DataFrame = pd.read_csv(full_result_path)
        artificial: pd.Series[bool] = self.data["feature"].str.contains(
            "artificial"
        )
        self.data["is_artificial"] = artificial
        start: int = self.path.index("log(lambda)_") + len("log(lambda)_")
        end: int = self.path.index("_to_")
        self.min_lambda = int(self.path[start:end])
        self._generate_top_exponent()
        self._generate_top_frequencies()
        self._make_wide_df()
        self._generate_fdr_results()
        self._generate_selected_features()

    def plot_stability_path(self, save: bool = False) -> None:
        """Method that plots stability paths"""
        _, ax = plt.subplots(figsize=(10, 6))
        self.wide_df.plot(ax=ax, legend=False)
        plt.xlabel(r"1 / λ")
        plt.ylabel(r"frequency (θ)")
        if save:
            fig_name: str = self.path + "_stability_path.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def plot_average_scores(self, save: bool = False) -> None:
        """Method that plots average C-index across bins"""
        assert len(self.data) > 1, "Results dict empty or 1"
        long_df = self.data
        sorted_indexes: np.ndarray = np.argsort(long_df["hyperparam"])
        plt.plot(
            long_df["hyperparam"][sorted_indexes],
            long_df["average_score"][sorted_indexes],
        )
        plt.xscale("log")
        plt.xlabel(r"$\lambda$")
        plt.ylabel("C-index")
        if save:
            fig_name = self.path + "_average_scores.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def plot_top_exponent(self, save: bool = False) -> None:
        """Plots the highest number of exponents which have frequencies
        higher than the highest artificial frequency

        Args:
            save (bool, optional): saves results in result_figs/.
            Defaults to False.
        """
        plt.plot(self.exponents, self.map_list)
        plt.axvline(
            self.top_exp,
            label=f"best exp: {round(self.top_exp, 4)}",
            ls=":",
            c="red",
        )
        plt.xlabel(r"log($\lambda$)")
        plt.ylabel("# non-artificial features")
        plt.legend()
        if save:
            fig_name = self.path + "_fdr_minimum.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def plot_top_freq_dist(self, save: bool = False) -> None:
        """Plots all top frequencies for both artificial and non-artificial
        features.

        Args:
            df (pd.DataFrame): result data fram from stability selection
            best_exp (float): exponent to 10, hyperparameter
        """

        sns.scatterplot(
            data=self.top_frequency_df,
            x="index",
            y="freq",
            hue="is_artificial",
        )
        if save:
            fig_name = self.path + "_frequency_distribution.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def plot_min_fdr(self, save: bool = False) -> None:
        """Calculates the false discovery rate
        (proxy) and plots agains cutoff to find the
        lowest possible fdr - known as the frequency threshold.
        Saves the frequency treshold as an instance variable.
        """
        min_fdr = min(self.row_df["fdr"])
        plt.plot(self.row_df["cut"], self.row_df["fdr"])
        plt.xlabel("cutoff")
        plt.ylabel("fdr")
        plt.axhline(
            min_fdr,
            label=f"min fdr: {round(min_fdr, 4)}",
            ls=":",
        )
        plt.axvline(
            self.reliability_thresh,
            ls=":",
            c="r",
            label="reliability thresh:"
            + f"{round(self.reliability_thresh, 4)}",
        )
        plt.legend()
        if save:
            fig_name = self.path + "_fdr_minimum.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def get_reliability_thresh(self) -> float:
        return self.reliability_thresh

    def plot_stability_path_with_thresh(self, save: bool = False) -> None:
        """Plots the stability paths with the frequency cutoff. Colors every
        10th non-chosen features grey.

        Args:
            save (bool, optional): set to True to save result in result_figs/.
            Defaults to False.
        """
        arbitrary_number = 200
        _, ax = plt.subplots(figsize=(10, 6))
        plt.axhline(
            self.reliability_thresh,
            label=f"reliability thresh: {round(self.reliability_thresh, 4)}\n"
            + f"no. of features: {len(self.selected_features)}",
            c="r",
            ls=":",
        )
        plt.legend()
        non_informative_df: pd.DataFrame = self.wide_df.drop(
            self.selected_features, axis=1
        )
        non_informative_df.iloc[:, :arbitrary_number].plot(
            ax=ax, legend=False, style=":", color="grey"
        )
        self.wide_df[self.selected_features].plot(ax=ax, legend=False)
        plt.xlabel(r"1 / λ")
        plt.ylabel(r"frequency (θ)")
        if save:
            fig_name: str = self.path + "_stability_path_w_thresh.pdf"
            plt.savefig(
                self.fig_path / fig_name,
                bbox_inches="tight",
            )
        plt.show()

    def get_selected_features(self) -> list[str]:
        return self.selected_features

    def get_top_frequencies(self) -> pd.DataFrame:
        return self.top_frequency_df

    def _true_feature_counter(self, exponent: int) -> int:
        """Helper function to count number of non-artificial features
        above max artificial feature frequency, runs in a loop with
        exponend iterable.

        Args:
            exponent (int): l1 exponent

        Returns:
            int: number of non-artificial features
        """
        filtered_df = self.data[self.data["hyperparam"] > 10**exponent]
        max_artificial = max(filtered_df[filtered_df["is_artificial"]]["freq"])
        selected_rows = filtered_df.query(f"freq > {max_artificial}")[
            "feature"
        ]
        unique_features: set[str] = set(selected_rows)
        true_features: int = len(unique_features)
        return true_features

    def _generate_top_exponent(self) -> None:
        """Runs upon initialization of plotting object to get data ready for
        plotting. By running this we allow for user to skip the vizualization
        step and go directly to plotting stability graphs. Saves instance
        variables exponents, map_list and top_exp.
        """
        self.exponents: np.ndarray = np.linspace(self.min_lambda, 0, 100)
        self.map_list: list[int] = list(
            map(self._true_feature_counter, self.exponents)
        )
        self.top_exp: float = self.exponents[
            self.map_list.index(max(self.map_list))
        ]

    def _generate_top_frequencies(self) -> None:
        """Filters out all data from hyperparams below best frequency, gets
        the indexes for the top frequency per feature, sorts the dataframe
        based on the indexes.
        By running this we allow for user to skip the vizualization
        step and go directly to plotting stability graphs, saves
        instance variable top_frequency_df.
        """
        df: pd.DataFrame = self.data[
            self.data["hyperparam"] > 10**self.top_exp
        ]
        idx: pd.Series[Any] = df.groupby("feature")["freq"].idxmax()
        max_rows: pd.DataFrame = df.loc[idx]
        max_rows.rename(columns={"Unnamed: 0": "index"}, inplace=True)
        max_rows["index"] = range(0, len(max_rows))
        self.top_frequency_df: pd.DataFrame = max_rows

    def _make_wide_df(self) -> None:
        """Helper function to make wide df instance variable"""
        df: pd.DataFrame = self.data[
            self.data["hyperparam"] > 10**self.top_exp
        ]
        wide_df: pd.DataFrame = df[
            df["hyperparam"] > 10**self.top_exp
        ].pivot(index="hyperparam", columns="feature", values="freq")
        wide_df = wide_df.copy()

        wide_df.index = 1.0 / wide_df.index.astype(float)
        wide_df = wide_df.sort_index()
        self.wide_df: pd.DataFrame = wide_df

    def _generate_fdr_results(self) -> None:
        rows: list[dict[str, float]] = []
        for cut in np.linspace(0, 1, 100):
            df_q = self.top_frequency_df.query(f"freq > {cut}")
            fdr = (
                1
                + df_q["feature"]
                .str.contains("artificial", case=False, na=False)
                .sum()
            ) / len(df_q["feature"])
            rows.append({"cut": cut, "fdr": fdr})

        row_df = pd.DataFrame(rows)
        thresh = float(
            row_df["cut"].iloc[list(row_df["fdr"]).index(min(row_df["fdr"]))]
        )
        self.row_df: pd.DataFrame = row_df
        self.reliability_thresh: float = thresh

    def _generate_selected_features(self) -> None:
        """Generates the selected features based on reliability thesh"""
        above_thresh_map: map[Any] = map(
            lambda x: max(self.wide_df[x]) > self.reliability_thresh,
            self.wide_df.columns,
        )
        selected_cols: pd.Index[str] = self.wide_df.iloc[
            :, list(above_thresh_map)
        ].columns
        self.selected_features: list[str] = list(selected_cols)


def _set_plt_params(remove_grid=False):
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    if remove_grid:
        sns.set_style("whitegrid", {"axes.grid": False})
    params = {
        "font.family": "DejaVu Serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium",
        "savefig.dpi": 300,
    }
    plt.rcParams.update(params)
    plt.style.use("tableau-colorblind10")

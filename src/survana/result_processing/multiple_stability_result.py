from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

from survana.config import PATHS
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)


class MultipleStabilityResult(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    features: dict[str, int] = Field(default_factory=dict[str, int])
    single_result_count: int = Field(default_factory=int)
    run_results: dict[float, dict[str, float | list[str]]] = Field(
        default_factory=dict
    )

    @classmethod
    def load_existing(cls, path: Path) -> BaseModel:
        with open(path, "r") as f:
            result_json = json.load(f)
        return cls.model_validate(result_json)

    def add_single_result(self, single_result: SingleStabilityResult) -> None:
        for feature in single_result.get_selected_features():
            if feature in self.features.keys():
                self.features[feature] += 1
            else:
                self.features[feature] = 1
        self.single_result_count += 1
        self._write_to_json_file(
            "accumulated_results_since_" + str(self.created_at)
        )

    def add_model_score(
        self,
        tuning_dict: dict[str, Any],
        features: list[str],
    ) -> None:
        best_score_index: int = np.argmax(tuning_dict["scores"])
        best_param: float = tuning_dict["params"]["alpha"][best_score_index]
        best_score: float = tuning_dict["scores"][best_score_index]  # ignore

        res: dict[str, Any] = {"best_param": best_param, "features": features}
        self.run_results.update({best_score: res})
        self._write_to_json_file(
            "accumulated_results_since_" + str(self.created_at)
        )

    def plot_top_score(self, save: bool = False) -> None:
        scores: list[float] = list(self.run_results.keys())
        hyp: list[float] = [
            best_param
            for score in scores
            for best_param in [self.run_results[score]["best_param"]]
            if isinstance(best_param, float)
        ]

        plt.plot(hyp, scores, "o")
        plt.xscale("log")
        plt.axhline(
            max(scores),
            label=f"top score {round(max(scores), 2)}",
            ls=":",
            c="r",
        )
        plt.ylabel("C-index")
        plt.xlabel(r"$\lambda$")
        if save:
            plt.savefig(
                PATHS["RESULT_FIGURES_DATA_PATH"]
                / f"multiple_stability_top_scores_{self.created_at}.pdf",
                bbox_inches="tight",
            )
        plt.legend()
        plt.show()

    def plot_feature_freq(
        self, cutoff_frequency: float = 0.4, save: bool = False
    ):
        features = self.get_top_features_with_frequencies(cut=cutoff_frequency)
        keys = list(features.keys())
        values = list(features.values())
        x = np.arange(len(keys))

        plt.bar(x, values, align="center")
        plt.xticks(x, keys, rotation=90)
        plt.ylabel("frequency")
        if save:
            plt.savefig(
                PATHS["RESULT_FIGURES_DATA_PATH"]
                / f"multiple_stability_freq_domain_{self.created_at}.pdf",
                bbox_inches="tight",
            )
        plt.show()

    def get_top_features_with_frequencies(
        self, cut: float = 0.5
    ) -> dict[str, float]:
        occurences: int = int(cut * self.single_result_count)
        sorted_feats: dict[str, float] = {
            k: v / self.single_result_count
            for k, v in sorted(
                self.features.items(), key=lambda item: item[1], reverse=True
            )
            if v > occurences
        }
        return sorted_feats

    def merge_results(self, other: MultipleStabilityResult) -> None:
        for feature, count in other.features.items():
            self.features[feature] = self.features.get(feature, 0) + count

        self.single_result_count += other.single_result_count
        self.run_results.update(other.run_results)

    def save_to_new_json(self, save_as="merged.json"):
        self._write_to_json_file(save_as)

    def _write_to_json_file(
        self, file_name: str, data_path: Path = PATHS["RESULT_JSON_DIR"]
    ) -> None:
        with open(data_path / file_name, "w") as f:
            f.write(self.model_dump_json())

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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

    def _write_to_json_file(
        self, file_name: str, data_path: Path = PATHS["RESULT_JSON_DIR"]
    ) -> None:
        with open(data_path / file_name, "w") as f:
            f.write(self.model_dump_json())

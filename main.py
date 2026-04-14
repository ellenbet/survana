import gc
import logging
from typing import Any

import numpy as np
import pandas as pd

from survana.config import CONFIG
from survana.data_processing.data_models import SksurvData
from survana.data_processing.dataloaders import (
    load_data_for_sksurv_coxnet,
    load_partial_data_for_sksurv_coxnet,
)
from survana.models.stability_selection import stability_selection
from survana.result_processing.multiple_stability_result import (
    MultipleStabilityResult,
)
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)
from survana.tuning.post_stability_selection import coxph_final_tuning_optuna

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet()

    accumulated_results = MultipleStabilityResult()
    max_round = CONFIG["tuning"]["mccv_splits"]
    for round in range(0, max_round):
        logger.info(f"Starting round {round} of {max_round}")
        single_result: SingleStabilityResult = stability_selection(
            data_collection=data_collection
        )

        assert (
            len(single_result.get_selected_features()) > 0
        ), "no features found, unacceptable error"
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
        del single_result
        del final_run_tuning
        del new_data_collection
        gc.collect()


if __name__ == "__main__":
    main()

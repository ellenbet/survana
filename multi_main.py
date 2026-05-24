import gc
import logging
from typing import Any

import numpy as np
import pandas as pd

from survana.config import CONFIG
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
from survana.tuning.post_stability_selection import cox

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

        cox_result = cox(sksurv=new_data_collection)
        logger.info("Starting Cox")
        logger.info(
            f"\nREGULAR COX RESULTS {cox_result} mean and std\n"
            f"top score: {np.mean(cox_result['scores'])}"
            + f" var: {np.var(cox_result['scores'])}"
            + f" std: {np.std(cox_result['scores'])}"
        )
        accumulated_results.add_single_result(single_result)
        accumulated_results.add_model_score(
            cox_result, features=single_result.get_selected_features()
        )
        del single_result
        del new_data_collection
        gc.collect()


if __name__ == "__main__":
    main()

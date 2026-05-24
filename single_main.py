import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survana.config import CONFIG, PATHS
from survana.data_processing.data_models import SksurvData
from survana.data_processing.dataloaders import (
    load_data_for_sksurv_coxnet,
    load_partial_data_for_sksurv_coxnet,
)
from survana.models.stability_selection import stability_selection
from survana.result_processing.plot_config import _set_plt_params
from survana.result_processing.single_stability_result import (
    SingleStabilityResult,
)
from survana.tuning.post_stability_selection import (
    cox,
    coxph_final_tuning_optuna,
    rsf_final_tuning,
)

_set_plt_params()
result_fig_path = PATHS["RESULT_FIGURES_DATA_PATH"]
result_fig_path.mkdir(parents=True, exist_ok=True)
log_file = result_fig_path / "post_tuning_results.txt"

logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(message)s",
    force=True,
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    data_collection: tuple[
        pd.DataFrame, pd.DataFrame, np.recarray[tuple[Any, ...], np.dtype[Any]]
    ] = load_data_for_sksurv_coxnet()

    single_result: SingleStabilityResult = stability_selection(
        data_collection=data_collection,
        version="stabl",
        override_min_lamda=-2.25,
    )

    if len(single_result.get_true_selected_features()) == 0:
        log_lambda_min = -2.25
        logger.info(f"Overriding log_lambda_min selection: {log_lambda_min}")
        single_result = SingleStabilityResult(
            full_result_path=single_result.full_result_path,
            denominator="stabl",
            n_lambdas=CONFIG["tuning"]["n_lambda"],
            overwrite_lambda_min=-log_lambda_min,
        )

    if len(single_result.get_true_selected_features()) == 0:
        log_lambda_min = -4.5
        logger.info(f"Overriding log_lambda_min selection: {log_lambda_min}")
        single_result = SingleStabilityResult(
            full_result_path=single_result.full_result_path,
            denominator="stabl",
            n_lambdas=CONFIG["tuning"]["n_lambda"],
            overwrite_lambda_min=-log_lambda_min,
        )

    logger.info(datetime.now())
    logger.info(PATHS["RESULT_CSV_DATA_PATH"])
    if len(single_result.get_selected_features()) == 0:
        return

    save = True
    show = False
    single_result.plot_top_exponent(save=save, show=show)
    single_result.plot_top_freq_dist(save=save, show=show)
    single_result.plot_min_fdr(save=save, show=show)
    single_result.plot_stability_path(save=save, show=show)
    single_result.plot_stability_path_with_thresh(save=save, show=show)
    new_data_collection: tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.recarray[tuple[Any, ...], np.dtype[Any]],
    ] = load_partial_data_for_sksurv_coxnet(
        single_result.get_selected_features()
    )
    logger.info("Selected features:")
    for feat in single_result.get_selected_features():
        logger.info(feat)

    print(
        "no. of features: "
        + str(len(single_result.get_selected_features()))
        + "\nreliability threshold: "
        + str(single_result.get_reliability_thresh())
        + "\nlambda_min: "
        + str(single_result.top_exp)
    )

    data = SksurvData(data_collection=new_data_collection)

    logger.info("Starting Cox-Ridge")
    cox_ph: dict[
        str, dict[str, Any] | np.ndarray[tuple[Any, ...], np.dtype[Any]]
    ] = coxph_final_tuning_optuna(
        data=data,
        model_type="ridge",
    )
    top_ind: int = np.argmax(cox_ph["scores"])
    logger.info(
        f"\nCOX-RIDGE RESULTS {cox_ph} mean val: {np.mean(cox_ph['scores'])} "
        f"best val with std and params:\n"
        f"{cox_ph['scores'][top_ind]}"  # type: ignore[index]
        f"{cox_ph['stds'][top_ind]}, "  # type: ignore[index]
        f"{cox_ph['params']['alpha'][top_ind]}"  # type: ignore[index]
    )

    rsf = rsf_final_tuning(data=data)
    logger.info("Starting RSF")
    logger.info(
        f"\nRANDOM SURVIVAL FORREST RESULTS {rsf}"
        f"mean val: {np.mean(rsf['scores'])} "
        f"best val with std and params:\n"
        f"{rsf['scores'][top_ind]},"
        f" {rsf['stds'][top_ind]}, {rsf['params'][top_ind]}"
    )

    cox_result = cox(sksurv=data)
    logger.info("Starting Cox")
    logger.info(
        f"\nREGULAR COX RESULTS {cox_result} mean and std\n"
        f"mean score: {np.mean(cox_result['scores'])}"
        + f" var: {np.var(cox_result['scores'])}"
        + f" std: {np.std(cox_result['scores'])}"
    )

    logger.info(
        "no. of true features: "
        + str(len(single_result.get_true_selected_features()))
        + "\nreliability threshold: "
        + str(single_result.get_reliability_thresh())
        + "\nlambda_min: "
        + str(single_result.top_exp)
    )

    single_res_row = {
        "folder_name": single_result.full_result_path,
        "data_input": PATHS["DATA_PATH"],
        "folds": CONFIG["tuning"]["rskf_splits"],
        "repeats": CONFIG["tuning"]["rskf_repeats"],
        "log_lambda_min": CONFIG["tuning"]["log_lambda_min"],
        "log_lambda_max": CONFIG["tuning"]["log_lambda_max"],
        "selected_log_lambda_min": single_result.get_top_exponent(),
        "n_true_features": len(single_result.get_true_selected_features()),
        "n_false_features": len(single_result.get_selected_features())
        - len(single_result.get_true_selected_features()),
        "fdp": single_result.get_min_fdp(),
        "reliability_threshold": single_result.get_reliability_thresh(),
        "cox_score": np.mean(cox_result["scores"]),
        "cox_ridge_score": cox_ph["scores"][np.argmax(cox_ph["scores"])],
        "rsf_score": rsf["scores"][np.argmax(rsf["scores"])],
        "cox_std": np.std(cox_result["scores"]),
        "cox_ridge_std": cox_ph["stds"][np.argmax(cox_ph["scores"])],
        "rsf_std": rsf["stds"][np.argmax(rsf["scores"])],
        "features_selected": ";".join(
            single_result.get_true_selected_features()
        ),
    }

    summary_path = Path(
        "final_2.5_only_2500_variance_single_stability_results.csv"
    )
    summary_df = pd.DataFrame([single_res_row])
    write_header = (
        not summary_path.exists() or summary_path.stat().st_size == 0
    )

    if not write_header:
        existing_columns = list(pd.read_csv(summary_path, nrows=0).columns)
        if existing_columns != list(summary_df.columns):
            raise ValueError(
                f"{summary_path} has columns {existing_columns}, but this "
                f"run would write {list(summary_df.columns)}. Rename or "
                "update the existing summary CSV before appending."
            )

    summary_df.to_csv(
        summary_path,
        mode="a",
        index=False,
        header=write_header,
    )


if __name__ == "__main__":
    main()

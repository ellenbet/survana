import numpy as np
import pandas as pd

from config import COEF_CUTOFF


def get_best_features(
    original_X: pd.DataFrame, all_coefs: np.ndarray
) -> dict[str, np.float64]:
    """Method that takes inn coefs found from model, sorts on largest
    absolute value and returns a dictionary with keys as features
    and values as coefficients

    Args:
        original_X (pd.DataFrame):
            Design matrix with feature column and named columns by feature
        all_coefs (np.ndarray):
            from model training, expects model.coefs_ from Sksurv package

    Returns:
        dict[str, np.float64]: keys as feature name, value as coefficient
    """
    indexed_coefs: list[tuple[int, np.float64]] = list(enumerate(all_coefs))
    sorted_coefs_with_index: list[tuple[int, np.float64]] = sorted(
        indexed_coefs,
        key=lambda x: abs(x[1]),  # type:ignore
        reverse=True,
    )[:COEF_CUTOFF]
    top_features: dict[str, np.float64] = {}
    for ind, coef_val in sorted_coefs_with_index:
        top_features[str(original_X.columns[ind])] = round(coef_val, 5)

    return top_features


# TODO make get_best_features for concatenated X

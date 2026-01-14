import logging
from typing import Any

import numpy as np
import pandas as pd
import sksurv.linear_model as lm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def robust_train(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray[tuple[Any, ...], np.dtype[np.float64]],
    param: float,
    train_ind: list[int],
    n_iter: int = 100,
) -> lm.CoxPHSurvivalAnalysis | lm.CoxnetSurvivalAnalysis | float:
    """Function to train on a specific parameter, with handling for
    singular matrix multiplication to prevent crash.

    Args:
        X (np.ndarray): design matrix
        y (np.ndarray): response variable
        param (float): param to train with
        train_ind (list[int]): train indexes

    Returns:
        model (lm.CoxPHSurvivalAnalysis): trained model
    """
    if model_type == "lasso":
        model = lm.CoxnetSurvivalAnalysis(
            alphas=[param], n_alphas=1, max_iter=1000
        )
    elif model_type == "ridge" or model_type == "ph":
        model = lm.CoxPHSurvivalAnalysis(
            alpha=param,
            n_iter=n_iter,
        )
    else:
        logger.error("No model type specified")
        return 0.0

    X_train = pd.DataFrame(X[train_ind, :])
    y_train: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = y[train_ind]

    try:
        model.fit(X_train, y_train)
    except np.linalg.LinAlgError:
        logger.error(
            "Singular matrix multiplication attempted - "
            + "skipping results.."
        )
        return 0.0
    except ArithmeticError:
        logger.error("Arithmetic error detected - skipping results..")
        return 0.0

    return model

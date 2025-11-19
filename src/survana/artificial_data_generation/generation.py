import numpy as np
import pandas as pd

from .methods import ArtificialType, mx_knockoffs, random_permutation


class ArtificialGenerator:
    """ArtificialGenerator class that calls upon functions
    in generation.py which produce artificial features,
    either model x knockoffs or random permutation.

    Attributes:
        n_artificial_features (int):
            number of artificial features,
            has to be less or equal to number of original features
        type (ArtificialType): either KNOCKOFF or RANDOM_PERMUTATION
        random_state (int): sets random state for generation

    Methods:
        fit_transform(X):
            Method that transforms any design matrix with n x m dimensions
            into a new design matrix with n x (m + n_artificial_features)
            dimension.



    """

    def __init__(
        self,
        n_artificial_features: int,
        artificial_type: ArtificialType,
        random_state: int | None = None,
    ) -> None:
        self.n_artificial_features: int = n_artificial_features
        self.type: ArtificialType = artificial_type
        self.random_state: int | None = random_state

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Method that transforms any design matrix with n x m dimensions
        into a new design matrix with n x (m + n_artificial_features)
        dimension.

        Args:
            X (pd.DataFrame): Design matrix

        Returns:
            X_concat (np.ndarray):
            Design matrix concatenated with artificial features
        """
        X_concat, self.articicial_indices = _make_artificial_features(
            X.values, self.n_artificial_features, self.type, self.random_state
        )
        return X_concat


def _make_artificial_features(
    X: np.ndarray,
    n_artificial_features: int = 0,
    artificial_type: ArtificialType = ArtificialType.KNOCKOFF,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function adjusted from Stabl:
    https://github.com/gregbellan/Stabl/blob/stabl_lw/stabl/stabl.py
    Function generating the artificial features.
    The artificial features will be concatenated to the original dataset.

    Args:
    X (np.ndarray): X.shape = (n_repeats, n_features), the input array

    artificial_type (ArtificialType) : defaults to KNOCKOFF

    n_artificial_features (int): number of artificial features to generate,
    defaults to 0, set to n_features if n_artificial = 0.

    Returns:
    X_out (np.ndarray):
        New design matrix with artificials,
        X.shape = (n_repeats, n_features + n_artificial)
    """
    n_features: int = X.shape[1]
    n_samples: int = X.shape[0]
    n_artificial: int = (
        n_features if not n_artificial_features else n_artificial_features
    )

    assert n_artificial <= n_features, (
        f"cannot have more artificial features ({n_artificial})"
        + f" than organic features ({n_features})."
    )

    if artificial_type == ArtificialType.RANDOM_PERMUTATION:
        X_permutation: np.ndarray = random_permutation(
            X, n_artificial, random_state
        )

    elif artificial_type == ArtificialType.KNOCKOFF:
        X_knockoff: np.ndarray = mx_knockoffs(
            X, n_samples, n_features, n_artificial, random_state
        )

    else:
        raise ValueError(
            "The type of artificial feature must be either"
            + f"{ArtificialType.KNOCKOFF} "
            + f"or {ArtificialType.RANDOM_PERMUTATION}."
            f" Got {artificial_type}"
        )

    X_artificial: np.ndarray = (
        X_permutation
        if artificial_type == ArtificialType.RANDOM_PERMUTATION
        else X_knockoff
    )
    X_concat: np.ndarray = np.concatenate([X, X_artificial], axis=1)
    artificial_indices: np.ndarray = np.arange(
        X_artificial.shape[1], X_concat.shape[1]
    )
    return X_concat, artificial_indices

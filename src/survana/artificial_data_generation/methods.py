from enum import StrEnum

import numpy as np
from knockpy.knockoffs import GaussianSampler
from matplotlib.pylab import Generator


class ArtificialType(StrEnum):
    """StrEnum class with two accepted types

    Attributes:
        KNOCKOFF (str): Set to "knockoff"
        RANDOM_PERMUTATION (str): Set to "permutation"
    """

    KNOCKOFF = "knockoff"
    RANDOM_PERMUTATION = "permutation"


def mx_knockoffs(
    X: np.ndarray,
    n_samples: int,
    n_features,
    n_artificial: int,
    random_state: int,
) -> np.ndarray:
    """Function adjusted from Stabl:
    https://github.com/gregbellan/Stabl/blob/stabl_lw/stabl/stabl.py
    mx_knockoffs function using knockpy functionality

    Args:
        X (np.ndarray): Design matrix
        n_samples (int): n
        n_features (_type_): p
        n_artificial (int): number of artificial features
        random_state (int): random state

    Returns:
        np.ndarray: Artificial features with shape (n, n_artificial)
    """
    np.random.seed(random_state)
    rng: Generator = np.random.default_rng(seed=random_state)
    if n_features > 3000:
        initial_shape: tuple[int, int] = (
            n_samples,
            (n_features // 3000 + 1) * 3000,
        )
        X_artificial: np.ndarray = np.empty(initial_shape)
        for i in range(n_features // 3000 + 1):
            cols: np.ndarray = rng.choice(
                a=n_features, size=3000, replace=False
            )
            X_tmp: np.ndarray = X[:, cols]
            X_art_tmp: np.ndarray = GaussianSampler(
                X_tmp, method="equicorrelated"
            ).sample_knockoffs()

            start: int = i * 3000
            stop: int = (i + 1) * 3000
            X_artificial[:, start:stop] = X_art_tmp
        X_artificial = X_artificial[
            :,
            rng.choice(
                a=X_artificial.shape[1], size=n_features, replace=False
            ),
        ]

    else:
        X_artificial = GaussianSampler(
            X, method="equicorrelated"
        ).sample_knockoffs()

    indices = rng.choice(
        a=X_artificial.shape[1], size=n_artificial, replace=False
    )
    X_artificial = X_artificial[:, indices]
    return X_artificial


def random_permutation(
    X: np.ndarray, n_artificial: int, random_state: int
) -> np.ndarray:
    """Function adjusted from Stabl:
    https://github.com/gregbellan/Stabl/blob/stabl_lw/stabl/stabl.py
    Generates n_artificial features through permutation.

    Args:
        X (np.ndarray): Ddesign matric
        n_artificial (int): amount of features to be generated
        random_state (int): random state

    Returns:
        np.ndarray: permutated design matrix to be combined with organic
    """
    rng: Generator = np.random.default_rng(seed=random_state)
    X_artificial: np.ndarray = X.copy()
    indices: np.ndarray = rng.choice(
        a=X_artificial.shape[1], size=n_artificial, replace=False
    )
    X_artificial = X_artificial[:, indices]
    for i in range(X_artificial.shape[1]):
        rng.shuffle(X_artificial[:, i])
    return X_artificial

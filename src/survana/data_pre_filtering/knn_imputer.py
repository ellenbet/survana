import pandas as pd
from sklearn.impute import KNNImputer

from survana.config import PATHS


def knn_imputer(
    prefiltered_datapth: str = str(PATHS["PREFILTERED_DATA_PATH_DOMAIN"]),
    sep=",",
    n_neighbors: int = 5,
    save: bool = False,
):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    X = pd.read_csv(prefiltered_datapth, index_col=0, sep=sep)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X.T).T, index=X.index, columns=X.columns
    )

    if save:
        X_imputed.to_csv(
            prefiltered_datapth.replace(".csv", "_knn-imputed.csv")
        )
    return X_imputed


if __name__ == "__main__":
    knn_imputer(save=True)

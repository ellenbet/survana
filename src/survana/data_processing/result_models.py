from typing import Any

import numpy as np


class Result:
    """Result class to compute frequencies per resulting
    np.array of coefficients found post test and train.

    Attributes:
        feature_names (np.ndarray | list[Any]):
            collection of feature names, length needs to match
            number of coefficients.
        rounding_cutoff (int):
            to be used in round() function, determines whether
            feature coefficient value is zero or non-zero, which
            translates to feature being selected or not.
        coefficient_number_cutoff (int):
            old variable, might remove. Determines maximum number
            of features allowed to be chosen. Defaults to None,
            resulting in all features being chosen.
        feature_frequencies (dict[str, list[int]]):
            Dict with each key being feature name and each value being
            a list of 0 to 1 that determines how many times each feature
            was selected during the process.

    """

    def __init__(
        self,
        feature_names: np.ndarray | list[Any],
        rounding_cutoff: int = 5,
        coefficient_number_cutoff: int | None = None,
    ) -> None:
        self.feature_frequencies: dict[str, list[int]] = {
            feature: [] for feature in feature_names
        }
        self.rounding_cutoff = rounding_cutoff
        self.coefficient_number_cutoff: int | None = coefficient_number_cutoff

    def save_frequencies(self, all_coefs: np.ndarray) -> None:
        """Method that saves feature selection frequency by defining all
        non-zero features as selected. Non-zero cutoff is defined by
        rounding_cutoff attribute.

        Args:
            all_coefs (np.ndarray): coefs from regression
        """
        for feature_name, coef_val in zip(
            self.feature_frequencies.keys(), all_coefs
        ):
            val: float = round(coef_val, self.rounding_cutoff)
            if val > 0.0:
                self.feature_frequencies[feature_name].append(1)
            else:
                self.feature_frequencies[feature_name].append(0)

    def get_top_features_and_save_frequencies(
        self,
        all_coefs: np.ndarray,
    ) -> dict[str, float]:
        """Method that takes inn coefs found from model, sorts on largest
        absolute value and returns a dictionary with keys as feature names
        and values as coefficients, removes all zero coefs. It is important to
        note that design_matrix_columns and all_coefs need to have
        corresponding indexes. Feature frequencies are also computed and saved
        in instance variable.

        Args:
            all_coefs (np.ndarray):
                from model training, expects model.coefs from Sksurv package

        Returns:
            dict[str, float]: keys as feature name, value as coefficient
        """
        design_matrix_columns: list[Any] = list(
            self.feature_frequencies.keys()
        )

        assert len(all_coefs) == len(design_matrix_columns), (
            f"Cannot get best features from design matrix X with rows:"
            f" {len(design_matrix_columns)} and coefficient "
            f"rows: {len(all_coefs)}"
            ", row numbers have to match in length and on indexes."
        )
        indexed_coefs: list[tuple[int, np.float64]] = list(
            enumerate(all_coefs)
        )
        sorted_coefs_with_index: list[tuple[int, np.float64]] = sorted(
            indexed_coefs,
            key=lambda x: abs(x[1]),  # type:ignore
            reverse=True,
        )[: self.coefficient_number_cutoff]
        top_features: dict[str, float] = {}

        for ind, coef_val in sorted_coefs_with_index:
            val: float = round(coef_val, self.rounding_cutoff)
            feature_name: Any = design_matrix_columns[ind]
            if val > 0.0:
                top_features[feature_name] = val
                self.feature_frequencies[feature_name].append(1)
            else:
                self.feature_frequencies[feature_name].append(0)

        return top_features

import numpy as np
import pandas as pd
import pytest

from survana.artificial_data_generation.generation import ArtificialGenerator
from survana.artificial_data_generation.methods import ArtificialType


@pytest.mark.parametrize(
    "art_type", [ArtificialType.RANDOM_PERMUTATION, ArtificialType.KNOCKOFF]
)
def test_generation(test_data: pd.DataFrame, art_type: ArtificialType):
    """Tests that data is created correctly and fake index preserved

    Args:
        test_data (pd.DataFrame): _description_
        art_type (ArtificialType): _description_
    """
    generator = ArtificialGenerator(
        n_artificial_features=3,
        artificial_type=art_type,
    )

    X_concat: np.ndarray = generator.fit_transform(test_data)
    assert X_concat.shape == (
        3,
        6,
    ), "Shape expected (3,6) for 3 artificial features"
    assert all(
        generator.articicial_indices == range(3, 6)
    ), "Expected artificial indices to be range(3, 6)"

    assert np.array_equal(
        test_data.values, X_concat[:, :3]
    ), "real data not found in concat X for expected indices"


@pytest.mark.parametrize(
    "art_type", [ArtificialType.RANDOM_PERMUTATION, ArtificialType.KNOCKOFF]
)
def test_artificial_proportion(
    test_data: pd.DataFrame, art_type: ArtificialType
):
    """Tests that artificial proportion does not exceed 1

    Args:
        test_data (pd.DataFrame): _description_
        art_type (ArtificialType): _description_
    """
    generator = ArtificialGenerator(
        n_artificial_features=4,
        artificial_type=art_type,
    )

    try:
        generator.fit_transform(test_data)
    except AssertionError:
        return

    assert 1 == 2, "proportion of artificial features > 1"

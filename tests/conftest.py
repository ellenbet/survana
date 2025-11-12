import pandas as pd
import pytest


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.read_csv("test_data/test_matrix.csv", index_col="tests")

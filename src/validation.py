import pandas as pd
import pytest
from src import preprocessing

def test_empty_dataframe():
    with pytest.raises(ValueError):
        preprocess(pd.DataFrame())

import pandas as pd
from src import preprocessing

def test_preprocessing():
    df = pd.DataFrame({
        "age": [25, 30],
        "salary": [50000, 60000],
        "churned": ["yes", "no"]
    })

    X, y = preprocessing(df)

    assert len(X) == 2
    assert len(y) == 2
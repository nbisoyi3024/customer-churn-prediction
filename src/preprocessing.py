import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(path):
     """ Load dataset from CSV """
     df = pd.read_csv(path)

#drop unnecessary columns
     df.drop(['RowNumber', 'CustomerId', 'Surname','Complain'], axis=1, inplace=True)

     X = df.drop('Exited', axis=1)
     y = df['Exited']

     return X,y


def preprocess_data(X,y):
    """
        Split dataset into training and testing sets
        (No feature engineering, no encoding here)
    """
    X_train, X_test, y_train, y_test = train_test_split(
                         X, y,
                                test_size=0.2,
                                random_state=42,
                                stratify=y
    )
    return X_train, X_test, y_train, y_test


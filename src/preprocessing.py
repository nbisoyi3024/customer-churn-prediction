import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(path = "/Users/niharikabisoyi/PyCharmMiscProject/Customer_churn /data/Customer-Churn-Records.csv"):
    """ Load dataset from CSV """
    df = pd.read_csv(path)

    #drop unnecessary columns
    df.drop(['RowNumber', 'CustomerId', 'Surname','Complain'], axis=1, inplace=True)

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    return X,y

def create_features(df):
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    sns.countplot(x='AgeGroup', hue='Exited', data=df)

    # credit score
    bins = [300, 500, 600, 700, 800, 900]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=False)
    sns.countplot(x='CreditScoreGroup', hue='Exited', data=df)

    #salary
    bins = [0, 30000, 60000, 100000, 200000]
    labels = ['Low', 'Medium', 'High', 'Vey High']
    df['SalaryGroup'] = pd.cut(df['EstimatedSalary'], bins=bins, labels=labels, right=False)
    sns.countplot(x='SalaryGroup', hue='Exited', data=df)

def preprocess_data(X,y):

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



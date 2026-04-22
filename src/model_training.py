from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def get_model_pipelines(X_train):
    """
    Create ML pipeline
    """
    categorical_cols = ["Geography", "Gender", "Card Type"]
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop='first',handle_unknown='ignore'), categorical_cols)
        ]
    )
    #Full pipeline
    pipelines = {
        "Random Forest": Pipeline([
                ("preprocessor",preprocessor),
                ("model", RandomForestClassifier(random_state=42))
            ]),
        "KNN": Pipeline([
                ("preprocessor",preprocessor),
                ("model",KNeighborsClassifier(n_neighbors=3))
            ]),
        "SVM" : Pipeline([
                ("preprocessor",preprocessor),
                ("model", SVC(
                       C=0.5,
                       kernel='rbf',
                       gamma='scale',
                       probability=True,
                       class_weight='balanced')
                 )
                 ]),
        "Gradient Boosting" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=3
                ))
            ])
           }
    return pipelines

def train_models(pipelines, X_train, y_train):
    """
       Train the models
       """
    trained_models = { }

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    return trained_models

def predict_models(models, X_test):
    """
    Generate predictions for all trained models
    """
    predictions = {}

    for name, model in models.items():
       # probability of churn (class 1)
        probs = model.predict_proba(X_test)[:, 1]

        predictions[name] = probs

    return predictions

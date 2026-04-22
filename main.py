#from sklearn.pipeline import Pipeline
import os
from src.preprocessing import  load_and_clean_data,preprocess_data
from src.model_training import get_model_pipelines, train_models
from src.model_evaluation import evaluate_models
from src.utils import save_model,load_model


DATA_PATH = "/Users/niharikabisoyi/PyCharmMiscProject/Customer_churn/data/Customer-Churn-Records.csv"
MODEL_PATH = "/Users/niharikabisoyi/PyCharmMiscProject/Customer_churn/models"

def run_pipeline():
        #Load & clean
        print("Loading and cleaning data...")
        X,y = load_and_clean_data(DATA_PATH)

        #Split data
        print("Preprocessing and splitting data...")
        X_train, X_test, y_train, y_test = preprocess_data(X,y)

        #Get pipelines
        print("Getting model pipelines...")
        pipelines = get_model_pipelines(X_train)

        #Train models
        print("Training models...")
        trained_models = train_models(pipelines,X_train, y_train)

        #Evaluate models
        print("Evaluating models...")
        results = evaluate_models(trained_models, X_test, y_test)

        print("\nModel Accuracy:")
        for model_name, metrics in results.items():
            print(f"\n{model_name} Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

        # Automatically select best model with max recall
        metric = "Recall"
        best_model = max(results, key=lambda x: results[x][metric])
        best_pipeline = trained_models[best_model]

        print(f"\nBest model based on {metric}: {best_model} ({results[best_model][metric]:.4f})")

        # Save the pipeline of the best model
        print("Saving the best model....")

        # save the best model as pipeline.pkl
        save_path = f"{MODEL_PATH}/pipeline.pkl"
        save_model(best_pipeline, save_path)

        # Load the best model pipeline
        best_pipeline = load_model(save_path)

        print("Pipeline completed successfully!")

        return results, trained_models,best_pipeline

print("Current working directory:", os.getcwd())

if __name__ == "__main__":
       run_pipeline()
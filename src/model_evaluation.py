from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate  multiple models using key classification metrics
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,  y_pred)
        precision = precision_score(y_test,  y_pred)
        recall = recall_score(y_test,  y_pred)
        f1 = f1_score(y_test,  y_pred)
        roc_auc = roc_auc_score(y_test,  y_pred)
        cl_report = classification_report(y_test,  y_pred)
        results[name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC_AUC_Score":roc_auc,
            "Classification Report": cl_report
        }

    return results


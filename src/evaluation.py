"""Test-set classification metrics with seizure as the positive class."""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_models(models, X_test, y_test):
    rows, predictions, matrices = [], {}, {}
    for name, model in models.items():
        predicted = model.predict(X_test)
        predictions[name] = predicted
        matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1])
        rows.append({'Model': name, 'Accuracy': accuracy_score(y_test, predicted),
                     'Precision': precision_score(y_test, predicted, zero_division=0),
                     'Recall': recall_score(y_test, predicted, zero_division=0),
                     'F1-score': f1_score(y_test, predicted, zero_division=0)})
    return pd.DataFrame(rows).set_index('Model'), predictions, matrices

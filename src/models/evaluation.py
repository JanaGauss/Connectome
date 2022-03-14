"""
Evaluation of fitted model on test data
"""
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, \
    mean_absolute_error, r2_score
import numpy as np
import pandas as pd


def model_evaluation(model, X_test, y_test):
    """
    Evaluates the model based on a set of metrics
    Classification: Accuracy, Precision, Recall, F1 and AUC
    Regression: MSE, MAE and R2
    checkout https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics for details
    Args:
        model: A fitted ML model
        X_test: The test dataset to be evaluated
        y_test: The true labels
        custom_metrics: A list of custom metrics
    Returns:
        Returns a dataframe containing model evaluations depending on prespecified metrics
    """
    if type(X_test) == list:
        assert X_test[0].shape[0] == len(y_test), 'X_test and y_test are not of equal length'
    else:
        assert len(X_test) == len(y_test), 'X_test and y_test are not of equal length'



    if len(np.unique(y_test)) == 2: # classification setting

        if model.__class__.__name__ in ['LogisticRegressionCV', 'RandomForestClassifier']:
          predictions = model.predict(X_test) # class labels
          score = model.predict_proba(X_test)[:, 1] # probabilities
        else:
          score = model.predict(X_test)
          predictions = np.round(score)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, score)

        return pd.DataFrame(
            {"Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "AUC": [auc]})
    else:
        predictions = model.predict(X_test)
        # regression setting
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return pd.DataFrame({"Mean Squared Error": [mse], "Mean Absolute Error": [mae], "Recall": [r2]})

import pandas as pd
import numpy as np
from typing import Any

def zero_one_loss(y: pd.Series, y_pred: np.ndarray) -> float:
    '''
    Evaluates the zero one loss

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values

    return: float - zero one loss
    '''
    return np.mean(y_pred != y)

def accuracy(y: pd.Series, y_pred: np.ndarray) -> float:
    '''
    Evaluates the accuracy

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values

    return: float - accuracy
    '''
    return np.mean(y_pred == y)

def precision(y: pd.Series, y_pred: np.ndarray, true: Any) -> float:
    '''
    Evaluates the precision

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values
    true: Any - the class of interest

    return: float - precision
    '''
    tp = np.sum((y_pred == true) & (y == true))
    fp = np.sum((y_pred == true) & (y != true))
    return tp / (tp + fp)

def recall(y: pd.Series, y_pred: np.ndarray, true: Any) -> float:
    '''
    Evaluates the recall

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values
    true: Any - the class of interest

    return: float - recall
    '''
    tp = np.sum((y_pred == true) & (y == true))
    fn = np.sum((y_pred != true) & (y == true))
    return tp / (tp + fn)

def f1_score(y: pd.Series, y_pred: np.ndarray, true: Any) -> float:
    '''
    Evaluates the f1 score

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values
    true: Any - the class of interest

    return: float - f1 score
    '''
    prec = precision(y, y_pred, true)
    rec = recall(y, y_pred, true)
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(y: pd.Series, y_pred: np.ndarray, true: Any) -> np.ndarray:
    '''
    Evaluates the confusion matrix

    y: pd.Series - true values
    y_pred: np.ndarray - predicted values
    true: Any - the class of interest

    return: np.np.ndarray - confusion matriy_pred
    '''
    tp = np.sum((y_pred == true) & (y == true))
    fp = np.sum((y_pred == true) & (y != true))
    fn = np.sum((y_pred != true) & (y == true))
    tn = np.sum((y_pred != true) & (y != true))
    return np.array([[tp, fp], [fn, tn]])

def print_report(y: pd.Series, y_pred: np.ndarray, true: Any) -> None:
    '''
    Prints the performance report: 0-1 loss, accuracy, precision, recall, f1 score, confusion matrix
    
    y: pd.Series - true values
    y_pred: np.ndarray - predicted values
    true: Any - the class of interest

    return: None
    '''
    print("0-1 Loss:", zero_one_loss(y, y_pred))
    print("Accuracy:", accuracy(y, y_pred))
    print("Precision:", precision(y, y_pred, true))
    print("Recall:", recall(y, y_pred, true))
    print("F1 Score:", f1_score(y, y_pred, true))
    print("Confusion Matriy_pred:")
    print(confusion_matrix(y, y_pred, true))
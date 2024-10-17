import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
from DecisionTree import DecisionTree
from RandomForest import RandomForest
import itertools
from joblib import Parallel, delayed

def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = .2, random_state: int = 123) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    X: pd.DataFrame, features
    y: pd.Series, target
    test_size: float, percentage of the dataset to include in the test split
    random_state: int, seed for the random number generator

    return: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], X_train, X_test, y_train, y_test
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    np.random.seed(random_state) 
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_index = int(num_samples * test_size) 
    train_indices = indices[split_index:]  
    test_indices = indices[:split_index]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

def evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, params: Dict[str, Any], model_class: Union[DecisionTree,RandomForest] ) -> float:
    '''
    Evaluate a model on the test set.

    X_train: pd.DataFrame - The input samples of the training set.
    y_train: pd.Series - The target values of the training set.
    X_test: pd.DataFrame - The input samples of the test set.
    y_test: pd.Series - The target values of the test set.
    params: Dict[str, Any] - The hyperparameters of the model.
    model_class: Union[DecisionTree, RandomForest] - The model class.

    Returns: float - The zero-one loss of the model.
    '''
    model = model_class(**params)
    model.fit(X_train, y_train)
    loss = model.zero_one_loss(X_test, y_test)
    return loss

def find_best_model(X: pd.DataFrame, y: pd.Series, results: List[Tuple[float, Dict[str, Any]]],  model_class: Union[DecisionTree, RandomForest]) -> Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float]:
    '''
    Finds the best model from the results of the hyperparameter grid search.

    X: pd.DataFrame - The input samples.
    y: pd.Series - The target values.
    results: List[Tuple[float, Dict[str, Any]]] - The list of zero-one losses and hyperparameters.
    model_class: Union[DecisionTree, RandomForest] - The model class.

    Returns: Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float] - The best model, the best hyperparameters, and the best zero-one loss.
    '''

    best_loss, best_params = min(results, key=lambda x: x[0])
    best_model = model_class(**best_params) # Once we have the best hyperparameters, we train the model on the entire dataset
    best_model.fit(X, y) 
    return best_model, best_params, best_loss



def hyperparameter_tuning(X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List], 
                          model: str = 'decision_tree', n_jobs: int = -1) -> Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float]:
    '''
    Finds the best model using the hyperparameter grid search.

    X: pd.DataFrame - The input samples.
    y: pd.Series - The target values.
    param_grid: Dict[str, List] - The hyperparameter grid to search over.
    model: str, default='decision_tree' - The type of the model. It can be 'decision_tree' or 'random_forest'.
    n_jobs: int, default=-1 - The number of jobs to run in parallel (-1 means using all processors).

    Returns: Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float] - The best model, the best hyperparameters, and the best zero-one loss.
    '''
    model_class = DecisionTree if model == 'decision_tree' else RandomForest
    param_combinations = list(itertools.product(*param_grid.values()))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model)(X_train, y_train, X_val, y_val, dict(zip(param_grid.keys(), params)), model_class) for params in param_combinations
    )
    results = list(zip(results, [dict(zip(param_grid.keys(), params)) for params in param_combinations])) 
    for result in results:
        print("Params: ", result[1], "Loss: ", result[0])
    return find_best_model(X, y, results, model_class)


def kfold(n_samples:int, cv: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Splits the data into training and test sets.

    n_samples: int - The number of samples.
    cv: int, default=5 - The number of splits.

    Returns: List[Tuple[np.ndarray, np.ndarray]] - The list of indices of the training and test sets.
    '''
    if cv < 2:
        raise ValueError('cv must be greater than 1')
    
    indices = np.random.permutation(n_samples) 
    fold_sizes = np.full(cv, n_samples // cv, dtype=int) # every fold has the same size
    fold_sizes[:n_samples % cv] += 1 # if the number of samples is not divisible by cv, the first n_samples % cv folds will have one more sample
    current = 0 
    folds = []

    for fold_size in fold_sizes: 
        start, stop = current, current + fold_size 
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]]) 
        folds.append((train_indices, test_indices))
        current = stop

    return folds

def cross_validation(X: pd.DataFrame, y: pd.Series, cv: int, params: Dict[str, Any], model_class: Union[DecisionTree, RandomForest]) -> float:
    '''
    Cross-validates the model.

    X: pd.DataFrame - The input samples.
    y: pd.Series - The target values.
    cv: int, default=5 - The number of splits for cross-validation.
    params: Dict[str, Any] - The hyperparameters of the model.
    model_class: Union[DecisionTree, RandomForest] - The model class.


    Returns: float - The average zero-one loss.
    '''
    folds = kfold(len(X), cv)
    losses = []
    
    for train_indices, test_indices in folds:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        loss = evaluate_model(X_train, y_train, X_test, y_test, params, model_class)
        losses.append(loss)
    mean_loss = np.mean(losses)
    print("Params: ", params, "CV Loss: ", mean_loss)
    return mean_loss

def hyperparameter_tuning_cv(X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List], model: str = 'decision_tree',
                             cv: int = 5, n_jobs: int = -1) -> Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float]:
    '''
    Finds the best model using the hyperparameter grid search and cross-validation.

    X: pd.DataFrame - The input samples.
    y: pd.Series - The target values.
    param_grid: Dict[str, List] - The hyperparameter grid to search over.
    model: str, default='decision_tree' - The type of the model. It can be 'decision_tree' or 'random_forest'.
    cv: int, default=5 - The number of splits for cross-validation.
    n_jobs: int, default=-1 - The number of jobs to run in parallel (-1 means using all processors).

    Returns: Tuple[Union[DecisionTree, RandomForest], Dict[str, Any], float] - The best model, the best hyperparameters, and the best zero-one loss.
    '''    
    model_class = DecisionTree if model == 'decision_tree' else RandomForest
    param_combinations = list(itertools.product(*param_grid.values()))
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(cross_validation)(X, y, cv, dict(zip(param_grid.keys(), params)), model_class) for params in param_combinations
    )
    results = list(zip(results, [dict(zip(param_grid.keys(), params)) for params in param_combinations]))

    return find_best_model(X, y, results, model_class)

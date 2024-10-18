import numpy as np
# from collections import Counter
from DecisionTree import DecisionTree
from performance import zero_one_loss
from typing import Tuple
import pandas as pd
from joblib import Parallel, delayed

class RandomForest:
    def __init__(self, n_trees:int=100, max_depth:int=None, min_samples_split:int=2, min_impurity_decrease:float=0.0, criterion:str='gini', max_features:str=None, n_jobs:int=1):
        '''
        n_trees: int, default=100 - The number of trees in the forest.
        max_depth: int, default=None - The maximum depth of the tree.
        min_samples_split: int, default=2 - The minimum number of samples required to split an internal node.
        min_impurity_decrease: float, default=0.0 - A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        criterion: str, default='gini' - The function to measure the quality of a split. Options are 'gini' and 'entropy'.
        max_features: str, default=None - The number of features to consider when looking for the best split. Options are None (all features), 'sqrt' (square root of the number of features), 'log2' (log base 2 of the number of features).
        n_jobs: int, default=1 - The number of jobs to run in parallel.
        '''
        if n_trees < 1:
            raise ValueError("n_trees must be greater than 0")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.max_features = max_features
        self.trees = []
        self.feature_names = None
        self.n_jobs = n_jobs

    def _bootstrap_sample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        '''
        Create a bootstrap sample of the data.
        
        X: pd.DataFrame - The input samples.
        y: pd.Series - The target values.

        return: Tuple[pd.DataFrame, pd.Series] - The bootstrap sample.
        '''
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Fit the random forest model.
        
        X: pd.DataFrame - The input samples.
        y: pd.Series - The target values.
        '''
        def _build_tree() -> Tuple[DecisionTree, pd.Index]:
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                min_impurity_decrease=self.min_impurity_decrease, criterion=self.criterion, max_features=self.max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            return tree, X_bootstrap.columns
        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_tree)() for _ in range(self.n_trees)
        )
        self.feature_names = X.columns

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predict the target values.

        X: pd.DataFrame - The input samples.

        return: np.ndarray - The predicted target values.
        '''
        tree_predictions = np.array([tree.predict(X[features]) for tree, features in self.trees])
        def most_common(arr):
            values, counts = np.unique(arr, return_counts=True)
            return values[np.argmax(counts)]

        majority_votes = [most_common(tree_predictions[:, i]) for i in range(X.shape[0])]
        return np.array(majority_votes)
    
    def zero_one_loss(self, X: pd.DataFrame, y: pd.Series) -> float:
        '''
        Evaluates the 0-1 loss

        X: pd.DataFrame - The input samples.
        y: pd.Series - The target values.

        return: float - 0-1 loss
        '''
        y_pred = self.predict(X)
        return zero_one_loss(y, y_pred)
    
    def feature_importance(self) -> np.ndarray:
        '''
        Calculate feature importances for the random forest.

        return: np.ndarray - The feature importances.
        '''
        importances = np.zeros(len(self.feature_names))
        for tree, feature_subset in self.trees:  
            tree_importances = tree.feature_importance()
            for i, feature in enumerate(feature_subset): 
                feature_idx = self.feature_names.get_loc(feature)  
                importances[feature_idx] += tree_importances[i]
        return importances / self.n_trees

    def print_importance(self) -> None:
        '''
        Prints the feature importances in descending order.
        '''
        features_importance = self.feature_importance()
        features_with_names = [(self.feature_names[idx], importance) for idx, importance in enumerate(features_importance)]
        features_with_names.sort(key=lambda x: x[1], reverse=True)
        print("Feature importance:")
        for feature, importance in features_with_names:
            print(f"{feature}: {importance}")
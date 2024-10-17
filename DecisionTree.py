import numpy as np
from joblib import Parallel, delayed
from typing import Optional, Any, Union, Tuple, List
import pandas as pd
from performance import zero_one_loss
import os
import subprocess
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature: Optional[int] = None, thr: Optional[Any] = None,
                left: Optional['Node'] = None, right: Optional['Node'] = None,
                value: Optional[Any] = None, categorical: bool = False):
        '''
        feature: int, default=None - The index of the feature to split on.
        thr: Any, default=None - The threshold to split the feature on.
        left: Node, default=None - The left child node.
        right: Node, default=None - The right child node.
        value: Any, default=None - The value of the leaf node.
        categorical: bool, default=False - Whether the feature is categorical.
        '''
        self.feature = feature
        self.thr = thr
        self.left = left
        self.right = right
        self.value = value
        self.categorical = categorical
        self.info_gain = None
        self.feature_name = None
        self.class_name = None
        self.criterion = None
        
    def is_leaf(self) -> bool:
        '''
        Returns: bool - True if the node is a leaf, False otherwise.
        '''
        return self.value is not None 

    def add_feature(self, info_gain: Optional[float] = None, feature_name: Optional[str] = None,
                    class_name: Optional[Any] = None) -> None:
        '''
        Adds information about the feature used to split the node.

        info_gain: Optional[float], default=None - The information gain of the node.
        feature_name: Optional[str], default=None - The name of the feature.
        class_name: Optional[Any], default=None - The class name.
        '''
        self.info_gain = info_gain
        self.feature_name = feature_name
        self.class_name = class_name
        
    def __str__(self) -> str:
        '''
        Returns: str - The string representation of the node.
        '''
        if not self.is_leaf():
            thr = f'{self.thr:.4f}' if not self.categorical else self.thr
            return f'Class: {self.class_name}\n {self.feature_name} {"=" if self.categorical else "<="} {thr} \n Info Gain: {round(self.info_gain, 4)} \n'
        return f'Class: {self.value}'

class DecisionTree:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, min_impurity_decrease: float = .0, criterion: str = 'gini', n_jobs: int = -1):
        '''
        max_depth: int, default=None - The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split: int, default=2 - The minimum number of samples required to split an internal node.
        min_impurity_decrease: float, default=0.0 - A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        criterion: str, default='gini' - The function to measure the quality of a split (either 'gini', 'scaled_entropy' or 'squared')
        n_jobs: int, default=-1 - The number of jobs to run in parallel. -1 means using all processors. 
        '''
        if max_depth is not None and max_depth < 1:
            raise ValueError('max_depth must be at least 1.')
        
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2.')
        
        if min_impurity_decrease < 0:
            raise ValueError('min_impurity_decrease must be at least 0.')
        
        if criterion not in ['gini', 'scaled_entropy', 'squared']:
            raise ValueError("Unsupported criterion. Use 'gini', 'scaled_entropy', or 'squared'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_jobs = n_jobs
        self.root = None
        self.min_gain = np.inf
        self.max_gain = -np.inf
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Fits the DecisionTree model to the training data.

        X: pd.DataFrame - The training input samples.
        y: pd.Series - The target values.
        '''
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X must be a DataFrame.')
        
        if not isinstance(y, pd.Series):
            raise ValueError('y must be a Series.')
        
        self.min_gain = np.inf
        self.max_gain = -np.inf
        self.n_features = X.shape[1]
        self.feature_importance_ = np.zeros(self.n_features)
        self.feature_types = self._get_feature_type(X)
        self.root = self._fit_rec(X, y)
        self.feature_names = X.columns

    def _get_feature_type(self, X: pd.DataFrame) -> List[str]:
        '''
        Returns the type of each feature in X: 'numerical' or 'categorical'.

        X: pd.DataFrame - The training input samples.
        
        Returns: List[str] - A list of feature types for each column in X.
        '''
        return ['numerical' if np.issubdtype(X[col].dtype, np.number) else 'categorical' for col in X.columns]

    def _fit_rec(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        '''
        Recursively builds the DecisionTree. This is an helper method for the fit method.

        X: pd.DataFrame - The training input samples.
        y: pd.Series - The target values.
        depth: int, default=0 - The current depth of the tree.

        Returns: Node - The root node of the tree.
        '''
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1): # If criteria are met or only one class is present, return a leaf node
            return Node(value=self._most_common_label(y))

        best_gain, best_feature, best_thr, is_categorical, left_idxs, right_idxs = self._best_split(X, y)

        if best_feature is None or len(left_idxs) == 0 or len(right_idxs) == 0 or best_gain < self.min_impurity_decrease: # If no split is found, return a leaf node or if one of the children is empty or the gain is too low
            return Node(value=self._most_common_label(y))
        
        feature_name = X.columns[best_feature]
        self.min_gain = min(self.min_gain, best_gain)
        self.max_gain = max(self.max_gain, best_gain)
        self.feature_importance_[best_feature] += best_gain
        # Recursively build left and right subtrees
        left = self._fit_rec(X.iloc[left_idxs], y.iloc[left_idxs], depth + 1) 
        right = self._fit_rec(X.iloc[right_idxs], y.iloc[right_idxs], depth + 1)
        node = Node(best_feature, best_thr, left, right, categorical=is_categorical)
        node.add_feature(best_gain, feature_name, self._most_common_label(y))
        return node

    def _best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[int],Optional[int], Optional[Any], bool, np.ndarray, np.ndarray]:
        '''
        Finds the best split for the current node. If no split is found, returns (None, None, False).

        X: pd.DataFrame - The training input samples.
        y: pd.Series - The target values.

        Returns: Tuple[Optional[int],Optional[int], Optional[Any], bool] - The index of the best gain, the index of the best feature, the best threshold, and whether the feature is categorical.
        '''
        # I use joblib to parallelize the search for the best split for each feature otherwise it would be too slow
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._best_split_for_feature)(X, y, feature) for feature in range(self.n_features)
            )
        best_gain, best_feature, best_thr, is_categorical, left_idxs, right_idxs = max(results, key=lambda x: x[0]) # Find the best split among all features
        return (best_gain, best_feature, best_thr, is_categorical, left_idxs, right_idxs) if best_gain > -np.inf else (None, None, None, False, [], []) # Return None if no split is found (i.e. all features are constant)

    def _best_split_for_feature(self, X: pd.DataFrame, y: pd.Series, feature: int) ->   Tuple[float, int, Any, bool, np.ndarray, np.ndarray]:
        '''
        Finds the best split for the given feature.

        X: pd.DataFrame - The training input samples.
        y: pd.Series - The target values.
        feature: int - The index of the feature to split on.

        Returns: Tuple[float, int, Any, bool, np.ndarray, np.ndarray] - The information gain, the index of the feature, the best threshold, whether the feature is categorical, the indices of the left and right child nodes.
        '''
        best_gain = -np.inf
        best_thr = None
        is_categorical = self.feature_types[feature] == 'categorical'
        X_column = X.iloc[:, feature] 
        left_idxs_best = []
        right_idxs_best = []

        if is_categorical:
            thrs = X_column.unique() 
        else:
            thrs = np.percentile(X_column, np.arange(10, 100, 10)) # 10-90th percentile to have less thresholds to search through
        
        for thr in thrs:
            gain, left_idxs, right_idxs = self._information_gain(X_column, y, thr, feature)
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                left_idxs_best = left_idxs
                right_idxs_best = right_idxs
        
        return best_gain, feature, best_thr, is_categorical, left_idxs_best, right_idxs_best

    def _information_gain(self, X_column: pd.Series, y: pd.Series, thr: Any, feature: int) -> Tuple[float, np.ndarray, np.ndarray]:
        '''
        Evaluates the information gain for the given feature and threshold.

        X_column: pd.Series - The column of the feature to split on.
        y: pd.Series - The target values.
        thr: Any - The threshold to split the feature on.
        feature: int - The index of the feature.

        Returns: Tuple[float, np.ndarray, np.ndarray] - The information gain, the indices of the left and right child nodes.
        '''
        if self.feature_types[feature] == 'categorical':
            left_idxs, right_idxs = self._split_categorical(X_column, thr)
        else:
            left_idxs, right_idxs = self._split_numerical(X_column, thr)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        if n_left == 0 or n_right == 0:
            return 0, [], []
        
        left_impurity, right_impurity = self._calculate_impurity(y.iloc[left_idxs]), self._calculate_impurity(y.iloc[right_idxs])
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        information_gain = self._calculate_impurity(y) - child_impurity
        return information_gain, left_idxs, right_idxs
    
    def _calculate_impurity(self, y: pd.Series) -> float:
        '''
        Evaluates the impurity of the target values.

        y: pd.Series - The target values.

        Returns: float - The impurity evaluated with the criterion.
        '''
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'scaled_entropy':
            return self._scaled_entropy(y)
        else:
            return self._squared_impurity(y)

    def _gini_impurity(self, y: pd.Series) -> float:
        '''
        Evaluates the Gini impurity of the target values. (2 * prob * (1 - prob))

        y: pd.Series - The target values.

        Returns: float - The Gini impurity.
        '''
        prob = y.value_counts(normalize=True)[0]
        return 2*prob*(1-prob)
    
    def _scaled_entropy(self, y: pd.Series) -> float:
        '''
        Evaluates the scaled entropy of the target values. (-prob/2 * log2(prob) - (1-prob)/2 * log2(1-prob))

        y: pd.Series - The target values.
        
        Returns: float - The scaled entropy.
        '''
        prob = y.value_counts(normalize=True)[0]
        if prob == 0 or prob == 1:
            return 0
        return -prob/2 * np.log2(prob) - (1-prob)/2 * np.log2(1-prob) 
    
    def _squared_impurity(self, y: pd.Series) -> float:
        '''
        Evaluates the squared impurity of the target values. (sqrt(prob * (1 - prob))

        y: pd.Series - The target values.
        

        Returns: float - The squared impurity.
        '''
        prob = y.value_counts(normalize=True)[0]
        return np.sqrt(prob * (1 - prob))

    
    def _split_numerical(self, X_column: pd.Series, split_thr: Union[int,float]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Splits the numerical feature on the given threshold.

        X_column: pd.Series - The column of the feature to split on.
        split_thr: Union[int,float] - The threshold to split the feature on.

        Returns: Tuple[np.ndarray, np.ndarray] - The indices of the left and right child nodes.
        '''
        X_column = X_column.to_numpy()
        left_idxs = np.argwhere(X_column <= split_thr).flatten()
        right_idxs = np.argwhere(X_column > split_thr).flatten()
        return left_idxs, right_idxs

    def _split_categorical(self, X_column: pd.Series, category: Any) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Splits the categorical feature on the given category.

        X_column: pd.Series - The column of the feature to split on.
        category: Any - The category to split the feature on.

        Returns: Tuple[np.ndarray, np.ndarray] - The indices of the left and right child nodes.
        '''
        left_idxs = np.argwhere(X_column == category).flatten()
        right_idxs = np.argwhere(X_column != category).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y: pd.Series) -> Union[Any]:
        '''
        Finds the most common label in the target values.

        y: pd.Series - The target values.

        Returns: Union[Any] - The most common label.
        '''
        return y.value_counts().index[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Predicts the target values for the input samples.

        X: pd.DataFrame - The input samples.

        Returns: np.ndarray - The predicted target values.
        '''
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X must be a DataFrame.')
        return np.array([self._predict_rec(x, self.root) for _, x in X.iterrows()])

    def _predict_rec(self, x: pd.Series, node: Node) -> Union[Any]:
        '''
        Recursively predicts the target value for the input sample. This is an helper method for the predict method.

        x: pd.Series - The input sample.
        node: Node - The current node in the tree.

        Returns: Union[Any] - The predicted target value.
        '''
        if node.is_leaf():
            return node.value
        
        if node.categorical:
            if x.iloc[node.feature] == node.thr:
                return self._predict_rec(x, node.left)
            return self._predict_rec(x, node.right)
        else:
            if x.iloc[node.feature] <= node.thr:
                return self._predict_rec(x, node.left)
            return self._predict_rec(x, node.right)

    def zero_one_loss(self, X: pd.DataFrame, y: pd.Series) -> float:
        '''
        Evaluates the accuracy of the model with 0-1 loss.

        X: pd.DataFrame - The input samples.
        y: pd.Series - The target values.

        Returns: float - The accuracy of the model evalueted with 0-1 loss.
        '''
        y_pred = self.predict(X)
        return zero_one_loss(y, y_pred)
    
    def feature_importance(self) -> np.ndarray:
        '''
        Returns the normalized feature importances.

        Returns: np.ndarray - The normalized feature importances.
        '''
        total_importance = np.sum(self.feature_importance_)
        if total_importance == 0:
            return np.zeros_like(self.feature_importance_)  # Return zeros if no importance
        return self.feature_importance_ / total_importance  # Normalize importances to sum to 1
    
    def print_importance(self) -> None:
        '''
        Prints the feature importances in descending order.
        '''
        features_importance = self.feature_importance()
        features_with_names = [(self.feature_names[idx], importance) for idx, importance in enumerate(features_importance)]
        features_with_names.sort(key=lambda x: x[1], reverse=True)
        print('Feature importance:')
        for feature, importance in features_with_names:
            print(f'{feature}: {importance}')

    def _get_color(self, info_gain) -> str:
        '''
        Returns the color based on the information gain.

        info_gain: float - The information gain.

        Returns: str - The hex color.
        '''
        norm_gain = np.clip((info_gain - self.min_gain) / (self.max_gain - self.min_gain), 0, 1)
        cmap = plt.cm.Blues
        # Adjust the range to avoid very light colors or very dark colors
        adjusted_norm_gain = 0.2 + (0.6 * norm_gain)  
        color = cmap(adjusted_norm_gain)
        return '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255)) # Convert to hex color

    def _dot(self) -> str:
        '''
        Returns the dot representation of the tree.

        Returns: str - The dot representation of the tree.
        '''
        dot = 'digraph {\n'
        dot += '\trankdir=TB;\n'
        dot += '\tnode [shape=ellipse];\n'

        def add_node(node: Node, parent_id: str) -> str:
            node_id = str(id(node))
            if node.info_gain is not None:
                color = self._get_color(node.info_gain) 
                dot = f'\t{node_id} [label="{str(node)}", style=filled, fillcolor="{color}"];\n'
            else:
                dot = f'\t{node_id} [label="{str(node)}", style=filled, fillcolor="#b2d1c6"];\n' 
        
            if parent_id != "":
                dot += f'\t{parent_id} -> {node_id};\n'

            if not node.is_leaf():
                left_dot = add_node(node.left, node_id)
                right_dot = add_node(node.right, node_id)
                dot += left_dot + right_dot
            return dot

        dot += add_node(self.root, "")
        dot += '}\n'
        return dot
    
    def draw_tree(self, filename: str) -> None:
        '''
        Saves the tree as a png file.

        filename: str - The name of the png file to save the tree.
        '''
        str_dot = self._dot()
        os.makedirs('imgs', exist_ok=True)
        command = f'dot -Tpng -o imgs/{filename}.png'
        subprocess.run(command, input=str_dot, text=True, shell=True, check=True)


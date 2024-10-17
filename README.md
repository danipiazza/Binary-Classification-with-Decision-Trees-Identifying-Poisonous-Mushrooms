# Binary Classification with Decision Trees: Identifying Poisonous Mushrooms

The aim of this project is to build a binary decision tree from scratch and then test it on a mushroom dataset. The goal is to predict whether a mushroom is poisonous or edible based on numerous attributes.

The first step was to analysed the data, divide it into training and test sets, before proceeding with a data cleaning phase.

The central point of the project was the construction of the decision tree for binary classification. Starting by creating a class for the nodes and then by developing the tree from the fit and prediction functions.

Three methods for the splitting criteria were used: Gini impurity, scaled entropy and squared error. As for the stopping criteria, three were implemented: maximum depth, minimum sample splitting and minimum impurity decrease. These criteria are essential to avoid underfitting or overfitting of the model.

To find the best criteria, the hyperparameters were adjusted, with and without cross-validation, using 0-1 loss as the evaluation metric. This step was essential to optimise the performance of the decision tree.

Finally, a random forest was constructed using the structures previously presented, in order to further study the predictive capabilities of the model and the importance of the features.

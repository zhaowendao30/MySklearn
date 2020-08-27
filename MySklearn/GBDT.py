import numpy as np
import progressbar
from utils.utils import accuracy_score, train_test_split, bar_widgets, to_categorical
from utils.loss import Cross_Entropy_Loss, Square_Loss
from DecisionTree import ClassificationTree, RegressionTree
from sklearn import datasets

class GradientBoosting():
    def __init__(self, n_estimators, learning_rate, min_samples_split, 
                min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.loss = Square_Loss()
        if not regression:
            self.loss = Cross_Entropy_Loss()
        
        self.trees = []

        for _ in range(n_estimators):
            tree = RegressionTree(min_impurity=min_impurity,
                                 min_samples_split=min_samples_split,
                                 max_depth=max_depth)
            self.trees.append(tree)
    
    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y, axis=0))
        for i in self.bar(range(self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # y_pred += learning_reate * (y - y_pred), 因为梯度为 y_pred - y，
            # 所以 y_pred -= learning_rate * (y_pred - y)
            y_pred -= np.multiply(self.learning_rate, update)
    
    def predict(self, X):
        y_pred = np.array([])

        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update
        
        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            return np.argmax(y_pred, axis=1)
    

class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5,
                    min_samples_split = 2, min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_impurity=min_var_red,
                    max_depth=max_depth,
                    regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5,
                    min_samples_split=2, min_var_red=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
        learning_rate=learning_rate, min_samples_split=min_samples_split,
        min_impurity = min_var_red, max_depth=max_depth, regression=False)
    
    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)


def main_classifier():
    
    print ("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(y_train.shape)

    clf = GradientBoostingClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


if __name__ == '__main__':
    main_classifier()

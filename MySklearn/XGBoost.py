from utils.utils import accuracy_score, train_test_split, bar_widgets, to_categorical
from DecisionTree import DecisionTree, ClassificationTree, RegressionTree
from utils.loss import Logistic_Loss
from sklearn import datasets
from tqdm import tqdm   # 进度条库--用来显示进度条
import numpy as np


class XGBoostRegressionTree(DecisionTree):
    def _split(self, y):
        col = y.shape[1] >> 1
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y_left, y_right):
        y, y_pred = self._split(y)
        y_left, y_left_pred = self._split(y_left)
        y_right, y_right_pred = self._split(y_right)

        gain = self._gain(y, y_pred)
        true_gain = self._gain(y_left, y_left_pred)
        false_gain = self._gain(y_right, y_right_pred)

        return true_gain + false_gain - gain
    
    def _appro_leaf(self, y):
        y, y_pred = self._split(y)

        gard = np.sum(self.loss.gradient(y, y_pred), axis=0)
        hess = np.sum(self.loss.hess(y, y_pred), axis=0)

        return gard/hess
    
    def fit(self, X, y):
        self.cal_impurity_value = self._gain_by_taylor
        self.cal_leaf_value = self._appro_leaf
        super(XGBoostRegressionTree, self).fit(X, y)


class XGBoost():
    def __init__(self, n_estimators=100, learning_rate=0.1, min_sample_split=2,
                min_impurity=1e-7, max_depth=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_sample_split = min_sample_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        
        self.loss = Logistic_Loss()

        self.trees = []
        for _ in range(self.n_estimators):
            tree = XGBoostRegressionTree(min_impurity=self.min_impurity,
                                        min_samples_split=self.min_sample_split,
                                        max_depth=self.max_depth,
                                        loss=self.loss)
            self.trees.append(tree)
    

    def fit(self, X, y):
        y = to_categorical(y)

        y_pred = np.zeros_like(y)

        for tree in tqdm(self.trees):
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_y_pred = tree.predict(X)

            y_pred -= np.multiply(self.learning_rate, update_y_pred)
    

    def predict(self, X):
        y_pred = None

        for tree in self.trees:
            update_y_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_y_pred)
            y_pred -= np.multiply(self.learning_rate, update_y_pred)
        
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

        return np.argmax(y_pred, axis=1)


def main():
    
    print ("-- XGBoost --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, seed=3)  

    clf = XGBoost(n_estimators=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)




if __name__ == "__main__":
    main()


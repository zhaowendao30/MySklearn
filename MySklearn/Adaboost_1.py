from DecisionTree import ClassificationTree
from utils.utils import accuracy_score, train_test_split
from sklearn import datasets
from tqdm import tqdm
import numpy as np

class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_i = None
        self.threshold = None
        self.alpha = None

class Adaboost_1():
    def __init__(self, n_clfs):
        self.n_clfs = n_clfs
        self.clfs = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        w = np.full(n_samples, (1 / n_samples))

        for _ in tqdm(range(self.n_clfs)):
            # 记录最小错误率的分类节点
            min_error = float('inf')
            # 实例化一个分类器
            clf = DecisionStump()
            # 寻找最佳分类特征和阈值    
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_feature = np.unique(feature_values)

                for threshold in unique_feature:
                    p = 1
                    
                    predictions = np.ones(y.shape)

                    predictions[X[:, feature_i] < threshold] = -1

                    error = sum(w[y != predictions])

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        clf.polarity = p
                        min_error = error
                        clf.feature_i = feature_i
                        clf.threshold = threshold
            clf.alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            negative_idx = (clf.polarity * X[:, clf.feature_i] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            w *= np.exp(-clf.alpha * y * predictions)
            w /= sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 1))

        for clf in self.clfs:
            predictions = np.ones((n_samples, 1))
            negative_idx = (clf.polarity * X[:, clf.feature_i] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions
        
        y_pred = np.sign(y_pred).flatten()

        return y_pred


def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # Change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost_1(n_clfs=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)


if __name__ == "__main__":
    main()


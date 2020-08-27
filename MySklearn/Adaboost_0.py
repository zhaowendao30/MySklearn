from DecisionTree import ClassificationTree
from utils.utils import accuracy_score, train_test_split
from sklearn import datasets
from tqdm import tqdm
import numpy as np

class Adaboost():
    def __init__(self, n_estimators = 5):
        self.clfs = []
        self.n_clfs = n_estimators
        self.alphas = [1 / n_estimators for _ in range(n_estimators)]
        self.polarity = [1 for _ in range(n_estimators)]
    def fit(self, X, y):
        n_samples, n_features = X.shape[0], X.shape[1]

        # 初始化各样本的权重
        w = np.full(n_samples, (1 / n_samples))
        # 储存每个分类器
        self.clfs = []

        for i in tqdm(range(self.n_clfs)):
            # 实例化一个分类器
            clf = ClassificationTree()
            # 训练
            clf.fit(X, y)
            # 得到训练结果
            y_pred = clf.predict(X)
            # 计算训练误差
            print(accuracy_score(y, y_pred))
            error = sum(w[y != y_pred])
            # 对于错误率大于0.5的分类器，因为是二分类问题(Adaboost只能解决二分类问题)，我们可以翻转
            # 分类器的预测结果来使得其错误率为1-error>0.5
            print(error)
            if error > 0.5:
                self.polarity[i] = -1
                y_pred *= -1
                error = 1- error
            self.alphas[i] = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            predictions = np.array(self.polarity[i] * y_pred)
            w *= np.exp(-self.alphas[i] * y * predictions)
            w /= sum(w)
            self.clfs.append(clf)
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 1))
        
        for i, clf in enumerate(self.clfs):
            predictions = np.array(self.polarity[i] * clf.predict(X))
            # 因为y_pred是er维的，所以要给predictions添加一维，否则不能传播
            predictions = np.expand_dims(predictions, axis=1)
            # 将每个分类器的结果按权重累加
            y_pred += (self.alphas[i] * predictions)
        
        # 利用符号函数得到结果并展平为一维的
        y_pred = np.sign(y_pred).flatten()

        return y_pred


def main():
    print('--- Adaboost ---')
    data = datasets.load_digits()
    X, y = data.data, data.target

    digit1 = 1
    digit2 = 8

    idx = np.append(np.where(y==digit1)[0], np.where(y==digit2)[0])
    y = data.target[idx]

    y[y==digit1] = 1
    y[y==digit2] = -1

    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = Adaboost(n_estimators=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)

    clf_tree = ClassificationTree()
    clf_tree.fit(X_train, y_train)
    y_pred_tree = clf_tree.predict(X_test)
    acc_tree = accuracy_score(y_pred, y_test)

    print("Adaboost_Accuracy:", acc)
    print("Tree_Accuracy:", acc_tree)

if __name__ == '__main__':
    main()


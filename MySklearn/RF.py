from utils.utils import train_test_split, accuracy_score, bar_widgets, random_split_data
from DecisionTree import DecisionTree, ClassificationTree, RegressionTree
from sklearn import datasets
from tqdm import tqdm
import numpy as np


class RandomForest():
    def __init__(self, n_estimators=10, feature_num=None, min_samples_split=2, min_impurity=1e-7,
                 max_depth=10):
        self.n_estimators = n_estimators                            # 决策树的数量
        self.min_samples_split = min_samples_split                  # 决策树生长所需的最小样本数量
        self.min_impurity = min_impurity                            # 最小不纯度--即最大纯度
        self.max_depth = max_depth                                  # 树的最大深度
        self.feature_num = feature_num                              # 每棵树所需的特征数量
        

        self.trees = []
        # 实例化每棵决策树
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_impurity=self.min_impurity,
                                      min_samples_split=self.min_samples_split,
                                      max_depth=self.max_depth)
            self.trees.append(tree)
        self.feature_index = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        # 将样本随机分为 n_estimators 个子集
        subsets = random_split_data(X, y, self.n_estimators)
        # 确定每棵树所需用到的特征数量，一般为总体特征的平方
        if self.feature_num is None:
            self.feature_num = int(np.sqrt(X.shape[1]))

        for i in tqdm(range(self.n_estimators)):
            # 取出每个子集的样本和标签
            X_i, y_i = subsets[i]
            # 随机选取特征，replace=False代表不重复选取，默认为True
            feature_idx = np.random.choice(np.arange(X_i.shape[1]), size=self.feature_num, replace=False)
            # 使得训练样本只包含我们所要用到的特征
            X_i = X_i[:, feature_idx]
            # 储存特征，预测时需要用到
            self.feature_index.append(feature_idx)
            # 训练决策树
            self.trees[i].fit(X_i, y_i)
    
    def predict(self, X):
        y_preds = np.zeros((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees):
            # 选取每棵树用到的特征
            feature_idx = self.feature_index[i]
            # y_preds每一行为每个样本对应的预测值，每一列对应每棵树的预测值
            y_preds[:, i] = tree.predict(X[:, feature_idx])
        
        y_pred = []
        for sample_pred in y_preds:
            # astype将float转为int，bincount方法直接受数组内元素为int
            label = np.bincount(sample_pred.astype('int')).argmax()
            y_pred.append(label)
        return np.array(y_pred)
            

def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)

    clf = RandomForest(n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
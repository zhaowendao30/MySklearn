from utils.utils import euler_distance, accuracy_score, train_test_split
from sklearn import datasets
import numpy as np

class KNN():
    def __init__(self, k):
        self.k = k
    
    def get_lable(self, neighbor_labels):
        # bincount数每个label的个数，并储存到count中，count[1]=3表示label==1的样本有3个
        count = np.bincount(neighbor_labels)
        return count.argmax()
    
    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])

        for i, temp in enumerate(X_test):
            # 计算得到距离temp欧拉距离最近的k个样本的下标
            index = np.argsort([euler_distance(temp, x) for x in X_train])[:self.k]
            # 根据下标得到距离temp欧拉距离最近的k个样本的标签
            neighbor_labels = y_train[index]
            # 根据最近的k个邻居的标签预测y_test的标签
            y_pred[i] = self.get_lable(neighbor_labels)
        return y_pred


if __name__ == "__main__":
    
    print ("-- KNN --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf = KNN(k=5)
    y_pred = clf.predict(X_test, X_train, y_train)
    
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)


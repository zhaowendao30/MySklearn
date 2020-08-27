import numpy as np
from utils.utils import accuracy_score, train_test_split, normalize, euler_distance
from sklearn import datasets

class KMeans():
    def __init__(self, k, max_iterations = 500):
        self.k = k
        self.max_iterations = max_iterations
    # 初始化聚类中心
    def init_center(self, X):
        n_samples, n_features = X.shape
        center_k = np.zeros((self.k, n_features))
        for i in range(self.k):
            center = X[np.random.choice(range(n_samples))]
            center_k[i] = center
        return center_k
    # 得到sample的最近的聚类中心
    def closest_center(self, sample, center_k):
        closest_center_i = 0                    # 最近的聚类中心
        closest_distance = float('inf')         # 离最近聚类中心的距离
        for i, center in enumerate(center_k):
            distance = euler_distance(sample, center)
            if distance < closest_distance:
                closest_distance = distance
                closest_center_i = i
        return closest_center_i
    # 进行聚类
    def create_clusters(self, center_k, X):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            center_i = self.closest_center(sample, center_k)
            # 将样本的下标储存到clusters中
            clusters[center_i].append(sample_i)
        return np.array(clusters)
    # 计算新的聚类中心
    def cal_cluster_center(self, X, clusters):
        n_features = X.shape[1]
        center_k = np.zeros((self.k, n_features))
        
        for i, cluster in enumerate(clusters):
            center = np.mean(X[cluster], axis=0)
            center_k[i] = center
        return center_k
    # 得到样本的类别
    def get_cluster_label(self, clusters, X):
        y_pred = np.zeros(X.shape[0])

        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        
        return y_pred
    # 样本X的类别
    def predict(self, X):
        center_k = self.init_center(X)

        for i in range(self.max_iterations):
            clusters = self.create_clusters(center_k, X)
            pre_center_k = center_k
            center_k = self.cal_cluster_center(X, clusters)

            diff = center_k - pre_center_k
            if not diff.any():
                break
        return self.get_cluster_label(clusters, X)

if __name__ == '__main__':
    X, y = datasets.make_blobs()

    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X)
    a = [y, y_pred]
    print(a)

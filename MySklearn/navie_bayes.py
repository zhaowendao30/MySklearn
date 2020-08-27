import numpy as np
import math
from sklearn import datasets
from utils.utils import accuracy_score, train_test_split


class Navie_Bayes():
    ''' 
    P(c|x) = (P(c)P(x|c))/P(x)
    假设x = (feature_0, feature_1, feature_2)有三个特征(属性),朴素贝叶斯算法假设各属性之间相互独立
    则P(x|c) = P(feature_0, feature_1, feature_2|c) = P(feature_0|c)*P(feature_1|c)*feature(feature_2|c)
    '''
    #计算得到 P(x|c)，储存到self.parameters中
    #self.parameters[c][feature_i] == P(feature_i|c)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = {}                                     # 储存 P(x|c) 或者密度函数的参数--均值和方差
        self.feature = [isinstance(x[0], float) for x in X.T]    # 储存特征的类别--是连续型还是离散型

        for c in self.classes:
            # 提取类别为c的样本
            X_where_c = X[np.where(y==c)]
            self.parameters[c] = {}
            for feature_i,col in enumerate(X_where_c.T):
                # 对于连续型数据的特征，我们使用密度函数计算他们的条件概率P(x|c)
                if self.feature[feature_i]:
                    parameters = {'mean': col.mean(), 'var': col.var()}
                    self.parameters[c][feature_i] = parameters
                # 对于离散型数据的特征(或者字符型),我们使用频率计算条件概率P(x|c)
                else:
                    self.parameters[c][feature_i] = self.cal_prior_feature_c(feature_i, c)
    # 计算先验概率P(c)
    def cal_prior_c(self, c):
        return np.mean(self.y == c)

    # 计算连续型特征的密度函数的相关参数--一般认为连续型特征的密度函数为正态密度函数
    def cal_likelihood(self, mean, var, x):
        eps = 1e-4 # 避免除数为0
        coeff = 1.0 / np.sqrt(2 * math.pi * var + eps)
        exponent = np.exp(-np.power(x - mean, 2)/ (2*var+eps))
        return coeff * exponent
    
    def cal_prior_feature_c(self, feature_i, c):
        eps = 1e-4 # 拉普拉斯修正
        X_where = X[np.where(self.y==c)]
        unique_features = np.unique(X[feature_i])
        dic_feature = {}
        for feature in unique_features:
            dic_feature[feature] = np.mean(X[feature_i] == feature) + eps
        return dic_feature

    def get_label(self, sample):
        # 储存各类别的概率
        posteriors = []
        for c in self.classes:
            posterior = self.cal_prior_c(c)
            for feature_i, feature_value in enumerate(sample):
                if self.feature[feature_i]:
                    likelihood = self.cal_likelihood(self.parameters[c][feature_i]['mean'], \
                    self.parameters[c][feature_i]['var'], feature_value)
                    posterior *= likelihood
                else:
                    posterior *= self.parameters[c][feature_i]
            # 储存x为类别c的概率
            posteriors.append(posterior)
        # 返回概率最大的类别
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X_test):
        y_pred = [self.get_label(x) for x in X_test]
        return y_pred


if __name__ == '__main__':
    
    print ("-- Navie-Bayes --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf = Navie_Bayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

import numpy as np
from utils.utils import accuracy_score, train_test_split

# 计算mse--均方误差
def mean_squared_error(y_pred, y_test):
    mse = np.mean(np.power(y_pred - y_test, 2))
    return mse


class Loss():
    def loss(self, y_true, y_pred):
        return NotImplementedError()
    
    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class Cross_Entropy_Loss():
    def __init__(self):
        pass
    
    def loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -y * np.log(p) - (1- y) * log(1-p)
    
    def accuracy(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))
    
    def gradient(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(y / p) + (1 - y) / (1 - p)


class Square_Loss():
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)
    
    def gradient(self, y, y_pred):
        return y_pred - y



class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return (self.__call__(x))(1 - self.__call__(x))



class Logistic_Loss():
    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient
    
    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)
    # 一阶导数
    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return p - y
    # 二阶导数
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)




import numpy as np
import progressbar


bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]
# 计算信息熵
def calEntropy(Y):
    entropy = 0.0
    log2 = lambda x: np.log(x)/np.log(2)
    unique_labels = np.unique(Y)
    n = len(Y)

    for label in unique_labels:
        p = len(Y[Y == label])/n
        entropy -= p*log2(p)
    return entropy

# 计算信息增益
def calGain(y, y0, y1):
    EntD = calEntropy(y)
    p = len(y0) / len(y)
    Gain = EntD - p * calEntropy(y0) - (1 - p) * calEntropy(y1)
    return Gain

# 计算信息增益率
def calGainRatio(y, y0, y1):
    log2 = lambda x: np.log(x)/np.log(2)
    p = len(y0)/len(y)
    IV = -p * log2(p) - (1 - p) * log2(1 - p)
    
    return calGain(y, y0, y1)/IV

# 计算基尼指数
def calGiniIndex(y, y0, y1):
    p = len(y0)/len(y)
    Gini_Index = p * calGini(y0) + (1 - p) * calGini(y1)
    return Gini_Index

# 计算基尼值
def calGini(y):
    Gini = 1
    unique_labels = np.unique(y)

    for label in unique_labels:
        Gini -= (len(y[y==label])/len(y))**2
    # 因为基尼指数划分准则是选最小的，为了使它与其他两个准则一样，用1-Gini代替
    return (1-Gini) 

cal_impurity = {}
cal_impurity['ID3'] = calGain
cal_impurity['C4.5'] = calGainRatio
cal_impurity['CART'] = calGiniIndex

# 划分特征
def split_on_feature(data, feature_i, threshold=None):
    split_fuc = None # 构造划分函数
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_fuc = lambda sample: sample[feature_i] >= threshold
    else:
        split_fuc = lambda sample: sample[feature_i] == threshold
    XY0 = np.array([x for x in data if split_fuc(x)])     # 满足条件的样本集
    XY1 = np.array([x for x in data if not split_fuc(x)]) # 不满足条件的样本集

    return np.array([XY0, XY1])

# 计算得分
def accuracy_score(y_pre, y_true):
    acc = np.sum(y_pre == y_true, axis=0)/len(y_true)
    return acc

# 随机数据
def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)

    return X[index], y[index]

# 划分训练集和测试集
def train_test_split(X, y, test_size = 0.1, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed=seed)
    
    split_index = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:] 
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

# 计算方差
def cal_variance(X):
    x_mean = np.mean(X, axis = 0)
    num_samples = np.shape(X)[0]
    variance = (1 / num_samples) * np.diag((X - x_mean).T.dot(X - x_mean))
    return np.array(variance, dtype=float)

# 计算标准差
def cal_std(X):
    return np.sqrt(cal_variance(X))

# 计算协方差矩阵
def cal_cov(X, Y = None):
    if Y is None:
        Y = X
    x_mean = X.mean(0)
    y_mean = np.mean(Y, axis = 0)
    num_samples = X.shape[0]
    assert np.shape(Y)[0] == X.shape[0]
    cov = (1 / num_samples) * np.diag((X- x_mean).T.dot(Y - y_mean))
    return np.array(cov, dtype=float)

# 计算相关系数矩阵
def cal_corr(X, Y = None):
    if Y is None:
        Y = X
    x_std = np.expand_dims(cal_std(X), axis = 1)
    y_std = np.expand_dims(cal_std(Y), axis = 1)
    corr = cal_cov(X, Y) / (x_std.T.dot(y_std))
    return np.array(corr, dtype=float)

# 计算欧拉距离
def euler_distance(x1, x2):
    euler_d = np.sqrt(np.sum(np.power(x1 - x2, 2)))
    return euler_d

def normalize(X, ord = 2, axis = -1):
    # 计算X的二范数，并使其维度至少为1,(即若l2为一个数字，np.atleast_1d可以给他增加一维，
    # 相当于np.expand_dims(l2, axis=1))
    l2 = np.atleast_1d(np.linalg.norm(X, ord = ord, axis = axis))
    # 使得l2中的0转化为1，方便后续归一操作(除数不能为0)
    l2[l2==0] = 1
    return X / np.expand_dims(l2, axis=axis)


def standardize(X):
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for col in range(X.shape[1]):
        # 方差为0的属性，将其设为0
        X_std[:, col] = (X[:, col] - mean[col]) / std[col] if std[col] else 0
    return X_std


def to_categorical(y, n_col = None):
    if not n_col:
        n_col = np.max(y) + 1
    one_hot = np.zeros((y.shape[0], n_col))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

    
def random_split_data(X, y, n_estimators, low=0.5, high=0.5):
    n_samples = X.shape[0]
    subsets = []
    for _ in range(n_estimators):
        rate = float(np.random.uniform(low, high, 1))
        n = int(rate * n_samples)
        idx = np.random.choice(range(n_samples), n)
        subsets.append([X[idx], y[idx]])
    return subsets




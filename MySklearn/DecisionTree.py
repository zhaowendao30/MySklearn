import numpy as np
from utils.utils import split_on_feature, cal_impurity, accuracy_score, calEntropy, train_test_split, cal_variance
import progressbar
from utils.utils import bar_widgets, to_categorical
from utils.loss import Cross_Entropy_Loss, Square_Loss, mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt

# 二叉树数据结构
class DecisionTreeNode():
    def __init__(self, mode='ID3', feature_i=None, threshold = None,
                 value=None, left_branch=None, right_branch = None):
        self.mode = mode
        self.feature_i = feature_i                 # 划分属性
        self.threshold = threshold                 # 划分依据属性值
        self.value = value                         # 叶子节点值
        self.left_branch = left_branch             # 左子树--True--属性feature_i中值为threshold或者大于threshold的样本
        self.right_branch = right_branch           # 右子树--False--属性feature_i中值不为threshold或者小于threshold的样本


class DecisionTree():
    def __init__(self, min_impurity=1e-7, min_samples_split=2,
                 max_depth=float("inf"), loss=None):
        self.root = None                                 # 树的根节点
        self.min_impurity = min_impurity                 # 最小不纯度
        self.min_samples_split = min_samples_split       # 决策树生长所需最小样本
        self.max_depth = max_depth                       # 决策树最大深度
        self.loss = loss                                 # 损失函数
        self.cal_impurity_value = None                   # 分裂准则/划分准则
        self.cal_leaf_value = None                       # 叶子节点的预测值

    def fit(self, X, Y, loss = None):
        self.root = self._build_tree(X, Y)
        self.loss = None


    def _build_tree(self, X, Y, cur_depth=0):

        '''
        决策树停止生成的条件有三个：
        1.样本数小于最小分割样本
        2.深度大于最大深度
        3.样本全部属于同一类别
        4.样本所有属性取值不变(即该子样本集中每个样本的每个属性都相同)
        '''

        max_impurity = 0
        best_criterias = None
        best_subsets = None


        if len(np.shape(Y)) == 1:
            Y = np.expand_dims(Y, axis=1)
        
        XY = np.concatenate((X, Y), axis=1)
        
        n_samples, n_features = np.shape(X)
        # 满足决策树生长条件：样本数量大于最小划分样本数量， 树的深度小于等于最大深度， 样本集类别数不为1
        if n_samples >= self.min_samples_split and cur_depth <= self.max_depth:
            # 选择最佳划分属性
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_features = np.unique(feature_values)
                # 选择featur_i属性的最佳划分点
                for threshold in unique_features:
                    XY0, XY1 = split_on_feature(XY, feature_i, threshold)
                    
                    # 若该子集的feature_i特征全部为一个值时，选择其他属性
                    if len(XY0) > 0 and len(XY1) > 0:

                        Y0 = XY0[:, n_features:]
                        Y1 = XY1[:, n_features:]
                        cur_impurity = self.cal_impurity_value(Y, Y0, Y1)
                    
                        if cur_impurity > max_impurity:
                            max_impurity = cur_impurity
                            best_criterias = {'feature_i': feature_i, 'threshold': threshold}
                            # 满足条件的样本放在左子树，不满足的放在右子树
                            best_subsets = {
                                'left_X': XY0[:, :n_features],
                                'left_Y': XY0[:, n_features:],
                                'right_X': XY1[:, :n_features],
                                'right_Y': XY1[:, n_features:]
                            }
        
        if max_impurity > self.min_impurity:
            
            left_branch = self._build_tree(best_subsets['left_X'], best_subsets['left_Y'], cur_depth+1)
            right_branch = self._build_tree(best_subsets['right_X'], best_subsets['right_Y'], cur_depth+1)
            return DecisionTreeNode(feature_i=best_criterias['feature_i'], threshold=best_criterias['threshold'],
                                         left_branch=left_branch, right_branch=right_branch)

        leaf_val = self.cal_leaf_value(Y)

        return DecisionTreeNode(value=leaf_val)


    def pre_val(self, x, tree=None):
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]    
        branch = tree.right_branch

        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch
        
        return self.pre_val(x, branch)


    def predict(self, X):
        pre_y = [self.pre_val(x) for x in X]
        return np.array(pre_y)


    def print_tree(self, tree=None, indent= ' '):
        if tree is None:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        else:
            print ("%s:%s? " % (tree.feature_i, tree.threshold))

            print ("%sT->" % (indent), end="")
            self.print_tree(tree.left_branch, indent + indent)

            print ("%sF->" % (indent), end="")
            self.print_tree(tree.right_branch, indent + indent)


class ClassificationTree(DecisionTree):
    def calleaf(self, Y):
        y_pre = None
        max_lable_num = 0

        if len(Y) == 1:
            Y = np.expand_dims(Y, axis=1)
        
        labels = np.unique(Y)
        for label in labels:
            label_num = len(Y[Y==label])
            if label_num > max_lable_num:
                max_lable_num = label_num
                y_pre = label
            
        return y_pre
    
    def fit(self, X, Y):
        self.cal_impurity_value = cal_impurity['ID3']
        self.cal_leaf_value = self.calleaf
        super(ClassificationTree, self).fit(X, Y)


class RegressionTree(DecisionTree):
    def cal_variance_reduction(self, y, y1, y2):
        y_var = cal_variance(y)
        y1_var = cal_variance(y1)
        y2_var = cal_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        criterion = y_var - (frac_1 * y1_var + frac_2 * y2_var)
        return sum(criterion)
    
    def cal_leaf_value_value(self, y):
        value = np.mean(y, axis = 0)
        # value的形式为[[]],若长度为1,应返回一个值，而不是数组
        return value if len(value) > 1 else value[0]

    def fit(self, X, Y):
        self.cal_impurity_value = self.cal_variance_reduction
        self.cal_leaf_value = self.cal_leaf_value_value
        super(RegressionTree, self).fit(X, Y)


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

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)



def main():
    
    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)



if __name__ == '__main__':
    main_classifier()












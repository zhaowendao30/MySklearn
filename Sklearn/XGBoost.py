from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# ① 分类
digits = load_digits()
data = digits.data
target = digits.target
print(data.shape,target.shape)
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.1)

# 单棵决策树
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("单棵决策树分类结果如下：")
print("混淆矩阵:")
print(confusion_matrix(y_pred,y_test))
print("训练集分数：",clf.score(x_train,y_train))
print("验证集分数：",clf.score(x_test,y_test))

clf = XGBClassifier(n_estimators=100,
                    max_depth=10,
                             )
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("XGBoost分类器（多棵决策树）分类结果如下：")
print("混淆矩阵:")
print(confusion_matrix(y_pred,y_test))
print("训练集分数：",clf.score(x_train,y_train))
print("验证集分数：",clf.score(x_test,y_test))
# Author: Hong Sheng Liu
# Date: 2023-08-17
# Description: This is a baseline of Decision Tree Classifier for RUMEDECT
# Version: 1.0
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import (
#     StandardScaler,
#     Normalizer,
#     MinMaxScaler,
#     MaxAbsScaler,
# )
from sklearn.tree import DecisionTreeClassifier

# 从JSON文件中读取数据
with open(
    "/home/zsc/Yf/ML/ActivateLearning/Rumor/tempFiles/output2.json",
    "r",
    encoding="ascii",
) as file:
    data = json.load(file)
# 解析json数据并创建特征和标签列表
features = []
labels = []


for entry in data:
    feature = entry["feature"]
    label = int(entry["label"])
    features.append(feature)
    labels.append(label)

# 创建数据集
dataset = pd.DataFrame(features)
dataset["label"] = labels
X = dataset.drop("label", axis=1)
y = dataset["label"]
# scaler = Normalizer()
# X_scaled = scaler.fit_transform(X)
y = np.array(y)
X = np.array(X)
# 准备数据集，假设您的特征数据保存在X变量中，目标变量保存在y变量中，这个是归一化后的结果
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

random_state_arry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy = []
precision = []
recall = []
f1 = []

# 这里要做10次不同训练，然后计算平均值
for i in range(10):
    # 这里是不需要归一化的结果
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state_arry[i]
    )

    # 创建逻辑回归模型
    model = DecisionTreeClassifier(criterion="gini")
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # print(model.predict_proba(X_test[0:10]))

    # print(y_test, y_pred)
    # 计算评估指标
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

# 打印评估指标
# print(accuracy)
# print(precision)
# print(recall)
# print(f1)

print("Accuracy:", np.mean(accuracy))
print("Precision:", np.mean(precision))
print("Recall:", np.mean(recall))
print("F1 Score:", np.mean(f1))

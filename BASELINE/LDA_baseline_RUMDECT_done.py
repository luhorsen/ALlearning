# Author: Hong Sheng Liu
# Date: 2023-06-29
# Version: 1.0
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler

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

# 对特征进行归一化
# 一个九个特征：Number_of_Senwords、Number_of_URL、Number_of_Comment、User_Type、
# RegisAge、Number_of_Followers、Number_of_posts、Number_of_reposts、Number_of_Followees
# 对于第四个特征User_Type（下标为3）是类别特征，不做归一化
# 如果不做归一化，会造成内存溢出；这里使用最大值、最小值，感觉效果最好
features_to_normalize = [
    "Number_of_Senwords",
    "Number_of_URL",
    "Number_of_Comment",
    "RegisAge",
    "Number_of_Followers",
    "Number_of_posts",
    "Number_of_reposts",
    "Number_of_Followees",
]
# scaler = Normalizer()
scaler = MinMaxScaler()
# X[features_to_normalize] = scaler.fit_transform(X[features_to_normalize])
X = scaler.fit_transform(X)
y = np.array(y)
X = np.array(X)
# 准备数据集，假设您的特征数据保存在X变量中，目标变量保存在y变量中

random_state_arry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy = []
precision = []
recall = []
f1 = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state_arry[i]
    )

    # 创建逻辑回归模型
    # model = MLPClassifier(hidden_layer_sizes=(128,), random_state=1)
    model = LinearDiscriminantAnalysis()

    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))


# 计算K-fold的平均评估指标
print("Accuracy:", np.mean(accuracy))
print("Precision:", np.mean(precision))
print("Recall:", np.mean(recall))
print("F1 Score:", np.mean(f1))

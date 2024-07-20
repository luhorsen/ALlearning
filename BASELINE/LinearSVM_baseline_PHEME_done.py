# Author: Hong Sheng Liu
# Date: 2023-06-29
# Version: 1.0
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler

# 从JSON文件中读取数据
with open(
    "/home/zsc/Yf/ML/ActivateLearning/Rumor/dataset/dataset.json",
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
scaler = MinMaxScaler()
# X[features_to_normalize] = scaler.fit_transform(X[features_to_normalize])
X = scaler.fit_transform(X)

# # 计算K-fold评估指标
# k = 5
# kf = KFold(n_splits=k)

random_state_arry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy = []
precision = []
recall = []
f1 = []

for i in range(10):
    # 这里是不需要归一化的结果
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state_arry[i]
    )

    # 创建逻辑回归模型
    model = LinearSVC()
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

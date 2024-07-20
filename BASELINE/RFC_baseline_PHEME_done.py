# Author: Hong Sheng Liu
# Date: 2023-08-07
# Version: 1.0
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
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
    "Length_of_Characters",
    "Number_of_Words",
    "User_Registration_Age",
    "Number_Of_Followers",
    "Statuses_Count",
    "Number_Of_Friends",
]
scaler = MinMaxScaler()
# X[features_to_normalize] = scaler.fit_transform(X[features_to_normalize])
X = scaler.fit_transform(X)
# 计算K-fold评估指标
# k = 5
# kf = KFold(n_splits=k)

random_state_arry1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random_state_arry2 = [572, 193, 845, 427, 669, 238, 714, 392, 51, 966]
accuracy = []
precision = []
recall = []
f1 = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state_arry1[i]
    )

    # 定义一个随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100, random_state=random_state_arry2[i]
    )
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

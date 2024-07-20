# Author: Hong Sheng Liu
# Date: 2023-08-07
# Version: 1.0
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, Normalizer

# 从JSON文件中读取数据
with open('E:\Rumor\weibo\\rumdect\output2.json', 'r') as file:
    data = json.load(file)

# 解析json数据并创建特征和标签列表
features = []
labels = []

for entry in data:
    feature = entry['feature']
    label = int(entry['label'])
    features.append(feature)
    labels.append(label)

# 创建数据集
dataset = pd.DataFrame(features)
dataset['label'] = labels
X = dataset.drop('label', axis=1)
y = dataset['label']
scaler = Normalizer()
X_scaled = scaler.fit_transform(X)

# 计算K-fold评估指标
k = 5
kf = KFold(n_splits=k)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 定义一个随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)


# 计算K-fold的平均评估指标
avg_accuracy = sum(accuracy_scores) / k
avg_precision = sum(precision_scores) / k
avg_recall = sum(recall_scores) / k
avg_f1 = sum(f1_scores) / k
# 打印评估指标
print("K-fold Accuracy:", avg_accuracy)
print("K-fold Precision:", avg_precision)
print("K-fold Recall:", avg_recall)
print("K-fold F1 Score:", avg_f1)

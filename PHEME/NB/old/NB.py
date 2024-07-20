import copy
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

from alipy import ToolBox

# X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2,
#                            n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
#                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)


with open(
    "/home/zsc/Yf/ML/ActivateLearning/Rumor/codes/AL_Comparision/NB_Unc/output2.json",
    "r",
) as file:
    data = json.load(file)
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
# 特征正则化处理

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
X[features_to_normalize] = scaler.fit_transform(X[features_to_normalize])
# scaler = Normalizer()
X_scaled = scaler.fit_transform(X)
y = np.array(y)
# 准备数据集，假设您的特征数据保存在X变量中，目标变量保存在y变量中
X = np.array(X)  # 特征数据


alibox = ToolBox(X=X, y=y, query_type="AllLabels", saving_path="./NB_Unc")

# Split data
alibox.split_AL(
    test_ratio=0.3, initial_label_rate=0.003, split_count=10, all_class=True
)

# Use the default Logistic Regression classifier
# model = alibox.get_default_model()
model = GaussianNB()

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion("num_of_queries", 200)

# Use pre-defined strategy
uncertainStrategy = alibox.get_query_strategy(strategy_name="QueryInstanceUncertainty")
unc_result = []

for round_exp in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round_exp)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round_exp)

    # Set initial performance point
    model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = alibox.calc_performance_metric(
        y_true=y[test_idx], y_pred=pred, performance_metric="accuracy_score"
    )
    saver.set_initial_point(accuracy)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing any sklearn models with proba_predict method are ok
        select_ind = uncertainStrategy.select(
            label_ind, unlab_ind, model=model, batch_size=1
        )
        # or pass your proba predict result
        # prob_pred = model.predict_proba(x[unlab_ind])
        # select_ind = uncertainStrategy.select_by_prediction_mat(unlabel_index=unlab_ind, predict=prob_pred, batch_size=1)

        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(
            y_true=y[test_idx], y_pred=pred, performance_metric="accuracy_score"
        )

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    unc_result.append(copy.deepcopy(saver))

analyser = alibox.get_experiment_analyser(x_axis="num_of_queries")
analyser.add_method(method_name="uncertainty", method_results=unc_result)

print(analyser)
analyser.plot_learning_curves(
    title="Example of alipy", std_area=False, saving_path="./NB_Unc"
)

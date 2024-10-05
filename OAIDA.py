import pandas as pd

from causal_discovery.select1 import select
from causal_discovery.corradel import remove_linearly_related_features
import warnings
import math
import time
from memory_profiler import memory_usage
import psutil

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score

# import river, an online learning library

from river import forest
from river import imblearn
from river import preprocessing

# Import the online learning metrics and algorithms from the River library
from river import metrics
from river import stream
from river import ensemble
from river.drift import ADWIN
from river.drift.binary import DDM, EDDM

total_time = 0  # 在函数外部定义 total_time
start_time = time.time()
start_memory = psutil.Process().memory_info().rss


def get_memory_usage():
    mem_usage = memory_usage(-1, interval=1, timeout=1)
    return max(mem_usage) - min(mem_usage)


data = pd.read_csv("C:/Users/Flora_OvO/Desktop/data1.csv")

# shuffle决定样本顺序是否被打乱
X = data.drop(['Label'], axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, shuffle=False,
                                                    random_state=0)  # 划分训练集（10%）和测试集（90%）

train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

treatment_node = 'Label'

name1 = "ARF-EDDM model"
model1 = imblearn.RandomOverSampler((
        preprocessing.MinMaxScaler() |
        forest.ARFClassifier(n_models=3, drift_detector=EDDM())),
    desired_dist={0: 0.4, 1: 0.6})

name2 = "ARF-DDM model"
model2 = imblearn.RandomOverSampler((
        preprocessing.MinMaxScaler() |
        forest.ARFClassifier(n_models=3, drift_detector=DDM())),
    desired_dist={0: 0.4, 1: 0.6})

name3 = "SRP-EDDM model"
model3 = imblearn.RandomOverSampler((
        preprocessing.MinMaxScaler() |
        ensemble.SRPClassifier(n_models=3, drift_detector=EDDM(), warning_detector=EDDM())),
    desired_dist={0: 0.4, 1: 0.6})

name4 = "SRP-ADWIN model"
model4 = imblearn.RandomOverSampler((
        preprocessing.MinMaxScaler() |
        ensemble.SRPClassifier(n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN())),
    desired_dist={0: 0.4, 1: 0.6})


#  名称更换:OAIDA
def OAIDA(model1, model2, model3, model4, train_data, test_data, total_time):
    # Record the real-time accuracy of PWPAE and 4 base learners
    # global drift
    memory_usage_per_100_samples = 0
    mem_usage_start = memory_usage(-1, interval=1, timeout=1)  # Monitor memory usage at the start

    metric = metrics.Accuracy()
    metric1 = metrics.Accuracy()
    metric2 = metrics.Accuracy()
    metric3 = metrics.Accuracy()
    metric4 = metrics.Accuracy()

    metric_w1 = []
    metric_w2 = []
    metric_w3 = []
    metric_w4 = []
    # prequential
    counter = 0  # 初始化计数器
    prequential_accuracy = []
    true_labels = []
    predicted_labels = []

    i = 0
    drift = 0
    t = []
    m = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    yt = []
    yp = []

    eddm = EDDM()
    #    adwin =ADWIN()

    hat1 = model1
    hat2 = model2
    hat3 = model3
    hat4 = model4

    selected_features = select(train_data, treatment_node=treatment_node,
                               alpha=0.05, use_ci_oracle=False, graph_true=None, enable_logging=True,
                               ldecc_do_checks=False,
                               max_tests=2000)

    # 训练集上的学习
    train_data_selected = train_data[selected_features + ['Label']]

    for data_point, _ in stream.iter_pandas(train_data_selected):
        xi1 = {k: v for k, v in data_point.items() if k != 'Label'}
        yi1 = int(data_point.get('Label'))
        hat1.learn_one(xi1, yi1)
        hat2.learn_one(xi1, yi1)
        hat3.learn_one(xi1, yi1)
        hat4.learn_one(xi1, yi1)

    # 测试集上的预测
    test_data_selected = test_data[selected_features + ['Label']]

    for data_point, _ in stream.iter_pandas(test_data_selected):
        counter += 1  # 每处理一个实例，计数器加1
        xi = {k: v for k, v in data_point.items() if k != 'Label'}
        yi = int(data_point.get('Label'))

        y_pred1 = hat1.predict_one(xi)
        y_prob1 = hat1.predict_proba_one(xi)
        hat1.learn_one(xi, yi)

        y_pred2 = hat2.predict_one(xi)
        y_prob2 = hat2.predict_proba_one(xi)
        hat2.learn_one(xi, yi)

        y_pred3 = hat3.predict_one(xi)
        y_prob3 = hat3.predict_proba_one(xi)
        hat3.learn_one(xi, yi)

        y_pred4 = hat4.predict_one(xi)
        y_prob4 = hat4.predict_proba_one(xi)
        hat4.learn_one(xi, yi)

        if y_pred1 == yi:
            metric_w1.append(0)
        else:
            metric_w1.append(1)
        if y_pred2 == yi:
            metric_w2.append(0)
        else:
            metric_w2.append(1)
        if y_pred3 == yi:
            metric_w3.append(0)
        else:
            metric_w3.append(1)
        if y_pred4 == yi:
            metric_w4.append(0)
        else:
            metric_w4.append(1)

        # Record their real-time accuracy
        metric1 = metric1.update(yi, y_pred1)
        metric2 = metric2.update(yi, y_pred2)
        metric3 = metric3.update(yi, y_pred3)
        metric4 = metric4.update(yi, y_pred4)

        # Calculate the real-time window error rates of four base learners
        if i < 1000:
            e1 = 0
            e2 = 0
            e3 = 0
            e4 = 0
        else:
            e1 = sum(metric_w1[round(0.9 * i):i]) / len(metric_w1[round(0.9 * i):i])
            e2 = sum(metric_w2[round(0.9 * i):i]) / len(metric_w2[round(0.9 * i):i])
            e3 = sum(metric_w3[round(0.9 * i):i]) / len(metric_w3[round(0.9 * i):i])
            e4 = sum(metric_w4[round(0.9 * i):i]) / len(metric_w4[round(0.9 * i):i])

        ep = 0.001  # The epsilon used to avoid dividing by 0
        # Calculate the weight of each base learner by the reciprocal of its real-time error rate
        ea = 1 / (e1 + ep) + 1 / (e2 + ep) + 1 / (e3 + ep) + 1 / (e4 + ep)
        w1 = 1 / (e1 + ep) / ea
        w2 = 1 / (e2 + ep) / ea
        w3 = 1 / (e3 + ep) / ea
        w4 = 1 / (e4 + ep) / ea

        # Make ensemble predictions by the classification probabilities
        if y_pred1 == 1:
            ypro10 = 1 - y_prob1[1]
            ypro11 = y_prob1[1]
        else:
            ypro10 = y_prob1[0]
            ypro11 = 1 - y_prob1[0]
        if y_pred2 == 1:
            ypro20 = 1 - y_prob2[1]
            ypro21 = y_prob2[1]
        else:
            ypro20 = y_prob2[0]
            ypro21 = 1 - y_prob2[0]
        if y_pred3 == 1:
            ypro30 = 1 - y_prob3[1]
            ypro31 = y_prob3[1]
        else:
            ypro30 = y_prob3[0]
            ypro31 = 1 - y_prob3[0]
        if y_pred4 == 1:
            ypro40 = 1 - y_prob4[1]
            ypro41 = y_prob4[1]
        else:
            ypro40 = y_prob4[0]
            ypro41 = 1 - y_prob4[0]

        # Calculate the final probabilities of classes 0 & 1 to make predictions
        y_prob_0 = w1 * ypro10 + w2 * ypro20 + w3 * ypro30 + w4 * ypro40
        y_prob_1 = w1 * ypro11 + w2 * ypro21 + w3 * ypro31 + w4 * ypro41

        if (y_prob_0 > y_prob_1):
            y_pred = 0
            # y_prob = y_prob_0
        else:
            y_pred = 1
            # y_prob = y_prob_1

        # Update the real-time accuracy of the ensemble model
        metric = metric.update(yi, y_pred)

        i += 1  # 更新变量 i，表示处理了一个样本

        # Detect concept drift
        val = 0
        if yi != y_pred:
            val = 1
        in_drift_result = eddm.update(float(val))

        if in_drift_result.drift_detected & (i > 1000):
            print(f"Change detected at index {i}")
            drift = 1  # indicating that a drift occurs

        # If a drift is detected
        if drift == 1:
            x_new = X_test[round(0.9 * i):i]
            y_new = y_test[round(0.9 * i):i]

            # Relearn the online models on the most recent window data (representing new concept data)
            hat1 = imblearn.RandomOverSampler((
                    preprocessing.MinMaxScaler() |
                    forest.ARFClassifier(n_models=3, drift_detector=EDDM())),
                desired_dist={0: 0.4, 1: 0.6})
            # ARF-ADWIN
            hat2 = imblearn.RandomOverSampler((
                    preprocessing.MinMaxScaler() |
                    forest.ARFClassifier(n_models=3, drift_detector=DDM())),
                desired_dist={0: 0.4, 1: 0.6})

            hat3 = imblearn.RandomOverSampler((
                    preprocessing.MinMaxScaler() |
                    ensemble.SRPClassifier(n_models=3, drift_detector=EDDM(), warning_detector=EDDM())),
                desired_dist={0: 0.4, 1: 0.6})

            hat4 = imblearn.RandomOverSampler((
                    preprocessing.MinMaxScaler() |
                    ensemble.SRPClassifier(n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN())),
                desired_dist={0: 0.4, 1: 0.6})

            # 重新进行特征选择
            data_new = pd.concat([pd.DataFrame(x_new), pd.DataFrame(y_new)], axis=1)
            cleaned_data = remove_linearly_related_features(data_new)

            selected_features = select(cleaned_data, treatment_node=treatment_node,
                                       alpha=0.05,
                                       use_ci_oracle=False, graph_true=None, enable_logging=True, ldecc_do_checks=False,
                                       max_tests=100000)

            new_data_selected = data_new[selected_features + ['Label']]

            for data_point, _ in stream.iter_pandas(new_data_selected):
                xj = {k: v for k, v in data_point.items() if k != 'Label'}
                yj = int(data_point.get('Label'))
                hat1.learn_one(xj, yj)
                hat2.learn_one(xj, yj)
                hat3.learn_one(xj, yj)
                hat4.learn_one(xj, yj)

                if j == 1:
                    print(len(xj))
                    j = 0
            drift = 0
        j = 1

        t.append(i)
        m.append(metric.get() * 100)

        yt.append(yi)
        yp.append(y_pred)

        if counter % 100 == 0:  # 检查是否处理了100的倍数个实例
            prequential_accuracy = metric.get()  # 获取当前的Prequential accuracy
            print(f"Prequential accuracy after {counter} instances: {prequential_accuracy * 100:.2f}%")
            elapsed_time = time.time() - start_time  # 计算经过的时间
            print(f"Time elapsed after {counter} instances: {elapsed_time} seconds")
            memory_usage_per_100_samples += get_memory_usage()
            memory_usage_mb_hours = (memory_usage_per_100_samples / (10 ** 6)) * total_time / 3600
            print(f"Memory usage (in MB-hours) after {counter} samples: {memory_usage_mb_hours} MB-hours")

    recall = recall_score(yt, yp)
    tn_count = 0
    fp_count = 0
    for predicted, actual in zip(yp, yt):
        if actual == 0 and predicted == 0:
            tn_count += 1
        elif actual == 0 and predicted == 1:
            fp_count += 1

    # outputs
    mem_usage_end = memory_usage(-1, interval=1, timeout=1)  # Monitor memory usage at the end

    specificity = tn_count / (tn_count + fp_count)
    g_mean = math.sqrt(recall * specificity)
    print("Accuracy: " + str(round(accuracy_score(yt, yp), 4) * 100) + "%")
    print("Precision: " + str(round(precision_score(yt, yp), 4) * 100) + "%")
    print("Recall: " + str(round(recall_score(yt, yp), 4) * 100) + "%")
    print("F1-score: " + str(round(f1_score(yt, yp), 4) * 100) + "%")
    print("G-mean: " + str(round(g_mean, 4) * 100) + "%")
    print(metric1.get() * 100)
    print(metric2.get() * 100)
    print(metric3.get() * 100)
    print(metric4.get() * 100)
    return t, m, mem_usage_start, mem_usage_end


name = "Proposed OAIDA model"
t, m, mem_usage_start, mem_usage_end = OAIDA(model1, model2, model3, model4, train_data, test_data, total_time)

end_time = time.time()
end_memory = psutil.Process().memory_info().rss


print(f"Memory usage at the start: {max(mem_usage_start) - min(mem_usage_start)}")
print(f"Memory usage at the end: {max(mem_usage_end) - min(mem_usage_end)}")

# 计算程序的运行时间和内存使用（以RAM小时计）
total_time = end_time - start_time
memory_usage_in_bytes = end_memory - start_memory
memory_usage_in_ram_hours = (memory_usage_in_bytes * total_time) / (3600 * 10 ** 9)  # 将内存使用量转换为GB小时

print(f"Total time taken: {total_time} seconds")
print(f"Memory usage (in RAM-hours): {memory_usage_in_ram_hours} GB-hours")

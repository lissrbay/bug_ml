import numpy as np
from sklearn.metrics import top_k_accuracy_score


def bug_probability(y_pred):
    y_pred = y_pred.detach().numpy()
    preds = []
    for n in range(y_pred.shape[0]):
        pred = []
        for i in range(y_pred.shape[1]):
            pred.append(0 if y_pred[n][i][0] > y_pred[n][i][1] else y_pred[n][i][1])
        preds.append(pred)
    return np.array(preds)


def accuracy(y_true, y_pred, top_k=2):
    y_pred = bug_probability(y_pred)
    y_true = np.argmax(y_true, axis=1)
    top_k_acc_score = top_k_accuracy_score(y_true, y_pred, k=top_k, labels = np.arange(80))
    return top_k_acc_score


def check_code_embeddings(X, y):
    has_code = y.copy()
    samples_count = X.shape[0]
    report_length = X.shape[1]
    for n in range(samples_count):
        for i in range(report_length):
            has_code[n][i] = 0
            if np.sum(X[n][i]) != 0:
                has_code[n][i] = 1
    return has_code


def count_embeddings_before_buggy_method(y, has_code):
    first_occurance_of_code = []
    for n in range(y.shape[0]):
        for i in range(y.shape[1]):
            if has_code[n][i] == 1:
                first_occurance_of_code.append(i < np.argmax(y[n]))
                break
    percentile = 0
    if first_occurance_of_code:
        percentile = np.array(first_occurance_of_code).mean() * 100
    print("{:.2f}% of reports has code embeddings before buggy method".format(percentile))
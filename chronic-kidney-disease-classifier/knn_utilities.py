
#Classification using KNN
import numpy as np
from collections import Counter

def eucliden_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


def make_predictionsk(k, X_train, Y_train, X_test):
    Y_pred = []
    for x_test in X_test:
        euclidean_distances = [eucliden_distance(x_test, x_train) for x_train in X_train]
        sorted_indices  = np.argsort(euclidean_distances)
        k_indices = sorted_indices[:k]
        k_labels  = [Y_train[i].astype(int) for i in k_indices]
        most_label = Counter(k_labels).most_common(1)
        label  = most_label[0][0]
        Y_pred.append(label)
    return np.array([Y_pred])



def predict_accuracy(k, X_train, Y_train, X_test, Y_test):
    predict = make_predictionsk(k, X_train, Y_train, X_test)
    return predict, ((np.sum(predict == Y_test.T))/len(Y_test))

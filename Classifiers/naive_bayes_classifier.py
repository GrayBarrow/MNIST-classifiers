import numpy as np
from preprocessing import preprocess_for_naive_bayes
from visualization import plot_confusion_matrix


def train_and_evaluate_naive_bayes(X_train_raw, X_test_raw, y_train, y_test):
    X_train_bin, X_test_bin = preprocess_for_naive_bayes(X_train_raw, X_test_raw)

    likelihood = np.zeros((10, 28, 28))  # empty grid for each class
    prior = np.zeros(10)  # will store probability of each class
    # loop through classes
    for i in range(10):
        print(f"calculating probabilities for {i}")
        class_count = np.sum(y_train == i)
        prior[i] = class_count / len(y_train)
        # vectorize, sum across 0th axis, across images
        likelihood[i] = (np.sum((X_train_bin[y_train == i]), axis=0) + 1) / (class_count + 2)  # add laplace smoothing

    print("testing")
    y_pred = []
    for x in X_test_bin:
        # given an x, find probability for each class
        # use log so we can add and prob doesnt vanish
        log_prob = np.log(prior)
        for i in range(10):
            for j in range(28):
                for k in range(28):
                    # for each pixel, find probabilty of it either on or off
                    if x[j, k] == 1:
                        log_prob[i] += np.log(likelihood[i][j][k])
                    else:
                        log_prob[i] += np.log((1 - likelihood[i][j][k]))
        y_pred.append(np.argmax(log_prob))

    plot_confusion_matrix(y_test, y_pred)

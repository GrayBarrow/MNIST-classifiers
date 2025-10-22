import numpy as np
from preprocessing import preprocess_for_knn
from visualization import plot_confusion_matrix


def eval_KNN(X_train, y_train, X_test, K):
    """
    Predicts class of test set using KNN
    :param X_train: X training set
    :param y_train: labels of training set
    :param X_test: items to test
    :param K: number of nearest neighbors to consider
    :return: predictions for all test items
    """
    y_pred = []
    for i, x in enumerate(X_test):
        # calculate euclidian distance
        distances = np.sqrt(np.sum((x - X_train) ** 2, axis=1))
        # get indices of lowest K distances
        k_indices = np.argsort(distances)[:K]
        # get labels of k indices
        digits = y_train[k_indices]
        # get most common label
        y_pred.append(np.bincount(digits).argmax())
        # keep track of how much is done
        if i % 1000 == 0:
            print(f"Processed {i}/{len(X_test)} test samples")
    return np.array(y_pred)


def train_and_evaluate_knn(X_train_raw, X_test_raw, y_train, y_test):
    X_train, X_test = preprocess_for_knn(X_train_raw, X_test_raw)

    print("testing K=1")
    y_pred = eval_KNN(X_train, y_train, X_test, 1)
    plot_confusion_matrix(y_test, y_pred)

    print("testing K=3")
    y_pred = eval_KNN(X_train, y_train, X_test, 3)
    plot_confusion_matrix(y_test, y_pred)

    print("testing K=5")
    y_pred = eval_KNN(X_train, y_train, X_test, 5)
    plot_confusion_matrix(y_test, y_pred)

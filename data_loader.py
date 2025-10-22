import numpy as np

def split_dataset(ds_path, train_ratio=0.8):
    """
    Splits dataset into train and test sets.
    Randomly selects for each set.
    :param ds_path: path to dataset
    :param train_ratio: ratio of train set
    :return: four sets, the train and test sets for x and y
    """
    data = np.load(ds_path)
    X, y = data['X'], data['y']
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    upto = int(train_ratio*len(X))
    X_train, X_test = X[indices[:upto]], X[indices[upto:]]
    y_train, y_test = y[indices[:upto]], y[indices[upto:]]
    return X_train, X_test, y_train, y_test

def load_mnist(ds_path='mnist_dataset.npz'):
    return split_dataset(ds_path)

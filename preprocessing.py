import torch
from torch import nn
from torchvision import transforms
import numpy as np

def preprocess_for_knn(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    return X_train, X_test

def preprocess_for_naive_bayes(X_train, X_test):
    # normalize on [0, 1] and binarize
    X_train_normalized, X_test_normalized = X_train / 255.0, X_test / 255.0
    X_train_bin, X_test_bin = np.where(X_train_normalized > 0.5, 1, 0), np.where(X_test_normalized > 0.5, 1, 0)
    return X_train_bin, X_test_bin

def preprocess_for_nn(X_train, X_test, y_train, y_test):
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    #normalize x and onehot encode y
    X_train = transforms.Normalize(mean=X_train.mean(), std=X_train.std())(X_train)
    X_test = transforms.Normalize(mean=X_test.mean(), std=X_test.std())(X_test)
    y_train_onehot = nn.functional.one_hot(y_train, num_classes=10).float()
    y_test_onehot = nn.functional.one_hot(y_test, num_classes=10).float()
    return X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test

def preprocess_for_cnn(X_train, X_test, y_train, y_test):
    X_train = torch.from_numpy(X_train).unsqueeze(dim=1).float()
    X_test = torch.from_numpy(X_test).unsqueeze(dim=1).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    # normalize x
    X_train = transforms.Normalize(mean=X_train.mean(), std=X_train.std())(X_train)
    X_test = transforms.Normalize(mean=X_test.mean(), std=X_test.std())(X_test)
    return X_train, X_test, y_train, y_test

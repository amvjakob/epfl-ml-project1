# -*- coding: utf-8 -*-
"""Feature Engineering and Model Selection"""

import numpy as np
import math


def standardize(x, mean=None, std=None):
    """
    Standardizes a data matrix.

    :param x: data
    :param mean: mean used for standardization
    :param std: standard deviation used for standardization
    :return: standardized data
    """

    if not mean:
        mean = np.mean(x, axis=0)

    if not std:
        std = np.std(x, axis=0)

    return (x - mean) / std, mean, std


def expand_2(X):
    """
    Augments a matrix to the second degree

    :param X: data
    :return: data augmented to second degree
    """
    result = X.copy()
    result = np.c_[result, np.power(result, 2)]
    n, d = X.shape
    for j in range(d):
        print(j)

        new = result[:, j][:, np.newaxis] * result[:, j + 1:]
        if len(new) > 0:
            result = np.c_[result, new]

    return result


def remove_NaN_features(x, threshold=0.0):
    """
    Removes the feature if it has more than a certain threshold of -999 value

    :param x: data
    :param threshold: data
    """
    n, d = x.shape
    result = []
    X = x.copy()

    for j in range(d):
        # get examples where the feature j is NaN
        positions = X[:, j] == -999
        if np.mean(positions) < threshold:
            if not len(result):
                result = X[:, j]
            else:
                result = np.c_[result, X[:, j]]

    return result


def replace_NaN_by_mean(x):
    """
    Replaces the -999 values of x by the mean of that feature vector

    :param x: data
    """
    n, d = x.shape
    result = x.copy()

    for j in range(d):
        # get examples where the feature j is NaN
        positions = result[:, j] == -999
        if np.sum(positions) > 0:
            # replace NaN values by mean based on non-NaN examples
            result[positions] = np.mean(result[~positions, j])

    return result


def replace_NaN_by_median(x):
    """
    Replaces the -999 values of x by the median of that feature vector

    :param x: data
    """
    n, d = x.shape
    result = x.copy()

    for j in range(d):
        # get examples where the feature j is NaN
        positions = result[:, j] == -999
        if np.sum(positions) > 0:
            # replace NaN values by mean based on non-NaN examples
            result[positions] = np.median(result[~positions, j])

    return result


def remove_features(data, features, feats, verbose=False):
    """
    This function removes features from the data and the features list

    :param data: tX data
    :param features: list of all features from load_csv
    :param feats: array of strings containing the features we want to remove
    :param verbose: output list of features successfully removed
    :return: new data, new features
    """

    idx_to_remove = -1 * np.ones(len(feats))
    removed = []

    for i, feat in enumerate(feats):
        if feat in features:
            idx_to_remove[i] = features.index(feat)
            removed.append(feat)

    if verbose:
        print("Features removed:", *removed, sep='\n')

    return np.delete(data, idx_to_remove, 1), np.delete(features, idx_to_remove)


def binarize_undefined(data, features, feats, verbose=False):
    """
    Additive Binarization of NaNs in a database.

    Adds a feature whose value is 1 if the value is defined in wanted feature
    column (and 0 otherwise).

    :param data: data with NaN values
    :param feats: features to take into account for additive binarization
    :param verbose: output features that are successfully additively binarized
    :return: new data
    """

    done = []

    for i, feat in enumerate(feats):

        # check if wanted feature is in feature list
        if feat in features:
            # find index where to analyze feature
            idx_to_analyze = features.index(feat)

            # expand data with 1 where value is defined, 0 where value is undefined
            data = np.c_[data, data[:, 2 * (idx_to_analyze != -999) - 1]]

            # add feature name
            features.append(features[idx_to_analyze] + "_NAN_BINARIZED")
            done.append(feat)

    if verbose:
        print("Features for which additive binarization was performed:", *done, sep='\n')

    return data, features
            
            
def cross_validate(y, tx, classifier, ratio, n_iter):
    """
    Cross Validate

    Shuffles dataset randomly n_iter times, divides tx in train and
    test to compute accuracy.

    :param y: y
    :param tx: data
    :param classifier: classifier for model fitting
    :param train: train function (fitting function)
    :param predict: prediction function
    :param ratio: train/test ratio
    :param n_iter: number of iterations
    :return: accuracy
    """

    n, d = tx.shape
    n_train = math.floor(ratio * n)

    accuracy = np.zeros(n_iter)

    for i in range(n_iter):
        shuffle_indices = np.random.permutation(np.arange(n))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]

        train_y = shuffled_y[:n_train]
        train_tx = shuffled_tx[:n_train,:]

        test_y = shuffled_y[n_train:]
        test_tx = shuffled_tx[n_train:,:]

        classifier.fit(train_y, train_tx)
        y_pred = classifier.predict(test_tx)

        accuracy[i] = compute_accuracy(y_pred, test_y)

    return accuracy


def find_max_hyperparam(classifier, lambdas):
    """
    Find Max Hyperparam

    Finds optimal lambda_ hyperparameter (lowest loss).

    :param classifier: lambda classifier function
    :param lambdas: array of possible lambdas
    :return: optimal trio of lambda_, weight, loss
    """

    w_best = []
    loss_best = np.inf
    lambda_best = 0

    for lambda_ in lambdas:
        w, loss = classifier(lambda_)
        print("Testing hyperparameter value %f - loss: %.3f" % (lambda_, loss))
        if loss < loss_best:
            w_best = w
            loss_best = loss
            lambda_best = lambda_

    return lambda_best, w_best, loss_best


def compute_accuracy(ypred, yreal):
    """
    Compute Accuracy

    :param ypred: predicted y
    :param yreal: real y
    :return: elementwise accuracy
    """

    return np.sum(ypred == yreal) / len(yreal)


def log_1_plus_exp_safe(x):
    """
    Computes log(1+exp(x)) avoiding overflow/underflow issues

    :param x: input
    :return: log(1+exp(x))
    """
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out





# -*- coding: utf-8 -*-
"""Helpers"""

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

    for j in range(d):
        # get examples where the feature j is NaN
        positions = x[:, j] == -999
        if np.mean(positions) < threshold:
            if not result:
                result = x[:, j]
            else:
                result = np.c_[result, x[:, j]]

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


def remove_features_on_pct(data, features, pct_sup, verbose=False):
    """
    This function removes features from the data and the features list
    depending on the percentage of NaN they contain

    :param data: tX data
    :param features: list of all features from load_csv
    :param verbose: output list of features successfully removed
    :return: new data, new features, feats
    """
    perc = (data[:, :] <= -999.0).sum(axis=0) / data.shape[0] * 100
    feats = np.array(features)
    feats = feats[np.array(perc > pct_sup)]
    new_data, new_features = remove_features(tX, features, feats, verbose)

    return new_data, new_features, feats


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

# taken from labs
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

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

def cross_validate_kfold(y, x, classifier, k_fold):
    """
    Cross Validate with k fold, without shuffling dataset

    :param y: y
    :param tx: data
    :param classifier: classifier for model fitting
    :param train: train function (fitting function)
    :param predict: prediction function
    :param k_fold: numbers of folds chosen
    :return: accuracy
    """

    accuracies = []

    # Splitting indices in fold
    ind = build_k_indices(x, k_fold)

    # Computations for each split in train and test
    for i in range(0, k_fold):

        ind_sort = np.sort(ind[i])
        ind_opp = np.array(sorted(set(range(0, x.shape[0])).difference(ind_sort)))

        xtrain, xtest = x[ind_opp], x[ind[i]]
        ytrain, ytest = y[ind_opp], y[ind[i]]

        classifier.fit(ytrain, xtrain)
        y_pred = classifier.predict(xtest)

        accuracies.append(compute_accuracy(y_pred, ytest))

    return accuracies

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

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

def kernel_RBF(X1, X2, sigma=1):
    N1, D1 = np.shape(X1)
    N2, D2 = np.shape(X2)
    K = np.zeros((N1, N2))

    for i in range(0, N1):
        for j in range(0, N2):
            K[i, j] = np.sum((X1[i] - X2[j]) ** 2)

    return np.exp(-K / (2 * sigma ** 2))

def kernel_poly(X1, X2, p=2):
    N1, D1 = np.shape(X1)
    N2, D2 = np.shape(X2)
    return np.power(np.ones((N1, N2)) + X1 @ X2.T, p)

def kernel_predict(kernel_fun, y, X, Xtest, *args, lambda_=0):
    K = kernel_fun(X, X, *args)
    Ktest = kernel_fun(Xtest, X, *args)

    u = np.linalg.solve(K + lambda_ * np.eye(len(y)), y)

    return np.sign(Ktest @ u)


def model_comparison(classifier, y, x, k_fold):
    names = []
    result = []
    for model_name, model in classifier:
        score = np.array(cross_validation_kfold(model, y, x, k_fold))
        result.append(score)
        names.append(model_name)
        print_message = "%s: Mean=%f STD=%f" % (model_name, score.mean(), score.std())
        print(print_message)

    fig = plt.figure()
    fig.suptitle('Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(result)
    ax.set_xticklabels(names)
    plt.show()
    return result, names


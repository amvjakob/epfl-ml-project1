# -*- coding: utf-8 -*-
"""Functions needed for the project"""
import numpy as np


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


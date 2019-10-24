# -*- coding: utf-8 -*-
"""Models"""

import numpy as np
import math
import implementations

###########################
# Least squares 
###########################

def least_squares_loss_function(y, tx, w):
    """
    Least Squares loss function using MSE.

    :param y: target
    :param tx: data
    :param w: weights
    :return: loss, gradient
    """
    n, d = tx.shape

    # define error vector
    e = y - tx @ w

    # return loss and gradient
    return 1/(2*n) * e.T.dot(e), -1/n * tx.T @ e


def least_squares(y, tx):
    """
    Least Squares regression using normal equations.

    :param y: target
    :param tx: data
    :return: weight, loss
    """

    n, d = tx.shape
    w = np.zeros(d)

    # solve normal equations
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    # return weights and loss
    return w, 1/(2*n) * np.sum((y - tx @ w) ** 2)


def least_squares_GD(y, tx, initial_w, max_iters, gamma=0):
    """
    Linear Regression using gradient descent.

    :param y: target
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma
    :return: weights, loss
    """
    def loss_function(w):
        # compute the loss and gradient using all examples
        return least_squares_loss_function(y, tx, w)

    if gamma == 0:
        return GD_linesearch(loss_function, initial_w, max_iters)
    else:
        return GD(loss_function, initial_w, max_iters, gamma)

    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma=0):
    """
    Linear Regression using stochastic gradient descent.

    :param y: target
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma
    :return: weights, loss
    """

    def loss_function(w):
        # select a single example to compute the loss and the gradient
        ySGD = []
        xSGD = []
        for y_sgd, tx_sgd in batch_iter(y, tx, 1):
            ySGD = y_sgd
            xSGD = tx_sgd
            break
        return least_squares_loss_function(ySGD, xSGD, w)

    if gamma == 0:
        return GD_linesearch(loss_function, initial_w, max_iters)
    else:
        return GD(loss_function, initial_w, max_iters, gamma)


def ridge_regression(y, tx, lambda_):
    """
    Ridge Regression using normal equations.
    (Least Squares with L2 regularization)

    :param y: target
    :param tx: data
    :param lambda_: hyperparamter
    :return: weight, loss
    """

    n, d = tx.shape
    w = np.zeros(d)

    # solve normal equations
    w = np.linalg.solve(tx.T @ tx + n*lambda_ * np.eye(d), tx.T @ y)

    # return loss and gradient
    # return weight and loss
    return w, 1/(2*n) * np.sum((y - tx @ w) ** 2) + lambda_/2 * w.T.dot(w)


def lasso_regression(y, tx, lambda_, initial_w, max_iters):
    """
    Lasso regression using gradient descent 
    (Least Squares with L1 regularization, sparse)

    :param y: target
    :param tx: data
    :param lambda_: hyperparamter
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :return: weight, loss
    """
    def loss_function(w):
        # compute the loss and gradient using all examples
        return least_squares_loss_function(y, tx, w)

    return GD_L1(loss_function, initial_w, lambda_, max_iters)


###########################
# Logistic regression
###########################

def log_reg_loss_function(y, tx, w):
    """Logistic regression loss function

    :param y: y
    :param tx: data
    :param w: weights
    :return: loss, gradient
    """
    n, d = tx.shape
    yXw = y * (tx @ w)

    # compute the function value
    f = np.sum(np.log(1. + np.exp(-yXw)))

    # compute the gradient value
    g = tx.T @ (- y / (1. + np.exp(yXw)))

    return f, g


def logistic_regression(y, tx, initial_w, max_iters, gamma=0, verbose=False):
    """
    Logistic Regression using gradient descent or SGD

    :param y: answers
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma parameter
    :param verbose: print out information
    :return: weights, loss
    """
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    if gamma == 0:
        return GD_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return GD(loss_function, initial_w, max_iters, gamma, verbose=verbose)

    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma=0, verbose=False):
    """
    L2 Regularized Logistic Regression using gradient descent or SGD

    :param y: answers
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma parameter
    :param verbose: print out information
    :return: weights, loss
    """

    def loss_function(w):
        f, g = log_reg_loss_function(y, tx, w)

        # add regularization terms
        f += lambda_ / 2. * w.dot(w)
        g += lambda_ * w

        return f, g

    if gamma == 0:
        return GD_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return GD(loss_function, initial_w, max_iters, gamma, verbose=verbose)
    

def logistic_regression_sparse(y, tx, lambda_, initial_w, max_iters, verbose=False):
    """
    L1 Regularized Logistic Regression using gradient descent 

    :param y: target
    :param tx: data
    :param lambda_: hyperparamter
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :return: weight, loss
    """
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    return GD_L1(loss_function, initial_w, lambda_, max_iters, verbose=verbose)

    




# -*- coding: utf-8 -*-
"""Implementations"""
import numpy as np


###########################
# Least squares 
###########################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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

    return GD(loss_function, initial_w, max_iters, gamma)
    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
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
        index = np.random.randint(0, len(y))
        ySGD = y[index]
        xSGD = np.array([tx[index,:]])
        return least_squares_loss_function(ySGD, xSGD, w)

    return GD(loss_function, initial_w, max_iters, gamma)

    
def least_squares_loss_function(y, tx, w):
    """
    Least Squares loss function.

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


###########################
# Logistic regression
###########################


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic Regression using gradient descent or SGD

    :param y: answers
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma parameter
    :return: weights, loss
    """
    
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    return GD(loss_function, initial_w, max_iters, gamma)

    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    L2 Regularized Logistic Regression using gradient descent or SGD

    :param y: answers
    :param tx: data
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: gamma parameter
    :return: weights, loss
    """

    def loss_function(w):
        f, g = log_reg_loss_function(y, tx, w)

        # add regularization terms
        f += lambda_ / 2. * w.dot(w)
        g += lambda_ * w

        return f, g

    return GD(loss_function, initial_w, max_iters, gamma)
    
    
def log_reg_loss_function(y, tx, w):
    """Logistic regression loss function

    :param y: y
    :param tx: data
    :param w: weights
    :return: loss, gradient
    """
    
    yXw = y * (tx @ w)

    # compute the function value
    f = np.sum(np.log(1. + np.exp(-yXw)))

    # compute the gradient value
    g = tx.T @ (- y / (1. + np.exp(yXw)))

    return f, g
    

###########################
# Solvers and helpers
###########################

def GD(loss_function, w, max_iters, gamma):
    """
    Gradient descent
    :param loss_function: function to calculate loss
    :param w: initial weight vector
    :param max_iters: maximum number of iterations
    :param gamma: gradient descent parameter
    :return: weight, loss
    """

    # inital evaluation
    f, g = loss_function(w)
    evals = 0

    while True:
        # gradient descent step
        w_new = w - gamma * g

        # compute new loss values
        f_new, g_new = loss_function(w_new)
        evals += 1

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test stopping conditions
        if evals >= max_iters:
            break

    return w, f


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

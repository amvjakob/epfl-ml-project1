# -*- coding: utf-8 -*-
"""Implementations"""

import numpy as np
import math

def least_squares_GD(y, tx, initial_w, max_iters, gamma=0):
    """
    Linear Regression using gradient descent.

    :param y: answers
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
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma=0):
    """
    Linear Regression using stochastic gradient descent.

    :param y: answers
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
        return gradient_descent_linesearch(loss_function, initial_w, max_iters)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma)

def least_squares(y, tx):
    """
    Least Squares regression using normal equations.

    :param y: y
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

    :param y: answers
    :param tx: data
    :param lambda_: lambda parameter
    :return: weight, loss
    """

    n, d = tx.shape
    w = np.zeros(d)

    # solve normal equations
    w = np.linalg.solve(tx.T @ tx + n*lambda_ * np.eye(d), tx.T @ y)

    # return loss and gradient
    # return weight and loss
    return w, 1/(2*n) * np.sum((y - tx @ w) ** 2) + lambda_/2 * w.T.dot(w)

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
        return gradient_descent_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma, verbose=verbose)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma=0, verbose=False):
    """
        Regularized Logistic Regression using gradient descent or SGD

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
        return gradient_descent_linesearch(loss_function, initial_w, max_iters, verbose=verbose)
    else:
        return gradient_descent(loss_function, initial_w, max_iters, gamma, verbose=verbose)


def least_squares_loss_function(y, tx, w):
    n, d = tx.shape

    # define error vector
    e = y - tx @ w

    # return loss and gradient
    return 1/(2*n) * e.T.dot(e), -1/n * tx.T @ e
    
def least_squares_sparse(y, tx, lambda_, initial_w, max_iters):
    def loss_function(w):
        # compute the loss and gradient using all examples
        return least_squares_loss_function(y, tx, w)

    return gradient_descent_sparse(loss_function, initial_w, lambda_, max_iters)

def log_reg_loss_function(y, tx, w):
    n, d = tx.shape
    yXw = y * (tx @ w)

    # compute the function value
    f = np.sum(np.log(1. + np.exp(-yXw)))

    # compute the gradient value
    g = tx.T @ (- y / (1. + np.exp(yXw)))

    return f, g
    
def logistic_regression_sparse(y, tx, lambda_, initial_w, max_iters, verbose=False):
    def loss_function(w):
        return log_reg_loss_function(y, tx, w)

    return gradient_descent_sparse(loss_function, initial_w, lambda_, max_iters, verbose=verbose)

def gradient_descent(loss_function, w, max_iters, gamma, verbose=False, *args):
    """
    Gradient descent
    :param loss_function: function to calculate loss
    :param w: initial weight vector
    :param max_iters: maximum number of iterations
    :param gamma: gradient descent parameter
    :param verbose: Print output
    :param args: extra arguments
    :return: weight, loss
    """

    # set an optimality stopping criterion
    optimal_g = 1e-4

    # inital evaluation
    f, g = loss_function(w, *args)
    evals = 0

    while True:
        # gradient descent step
        w_new = w - gamma * g

        # compute new loss values
        f_new, g_new = loss_function(w_new, *args)
        evals += 1

        # print progress
        if verbose:
            print("%d - loss: %.3f" % (evals, f_new))

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test stopping conditions
        if np.linalg.norm(g, float('inf')) < optimal_g:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optimal_g)
            break

        if evals >= max_iters:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_iters)
            break

    return w, f


def gradient_descent_linesearch(loss_function, w, max_iters, verbose=False, *args):
    """
    Linesearch Gradient Descent.
    Uses quadratic interpolation to find best
    possible gamma value.
    :param loss_function: function to calculate loss
    :param w: initial weight vector
    :param max_iters: maximum number of iterations
    :param verbose: Print output
    :param args: extra arguments
    :return: weight, loss
    """

    # set an optimality stopping criterion
    linesearch_optTol = 1e-2

    # linesearch param
    linesearch_beta = 1e-4

    # evaluate the initial function value and gradient
    f, g = loss_function(w, *args)
    evals = 0
    gamma = 1.

    while True:
        # line-search using quadratic interpolation to
        # find an acceptable value of gamma
        gg = g.T.dot(g)
        w_evals = 0

        while True:
            # compute params
            w_new = w - gamma * g
            f_new, g_new = loss_function(w_new, *args)

            w_evals += 1
            evals += 1

            if f_new <= f - linesearch_beta * gamma * gg:
                # we have found a good enough gamma to decrease the loss function
                break

            if verbose:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # update step size
            if np.isinf(f_new):
                gamma = 1. / (10 ** w_evals)
            else:
                gamma = (gamma ** 2) * gg / (2. * (f_new - f + gamma * gg))

        # print progress
        if verbose:
            print("%d - loss: %.3f" % (evals, f_new))

        # update step-size for next iteration
        y = g_new - g
        gamma = -gamma * np.dot(y.T, g) / np.dot(y.T, y)

        # safety guards
        if np.isnan(gamma) or gamma < 1e-10 or gamma > 1e10:
            gamma = 1.

        # update weights / loss / gradient
        w = w_new
        f = f_new
        g = g_new

        # test termination conditions
        if np.linalg.norm(g, float('inf')) < linesearch_optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % linesearch_optTol)
            break

        if evals >= max_iters:
            if verbose:
                print("Reached maximum number of function evaluations %d" % max_iters)
            break

    return w, f

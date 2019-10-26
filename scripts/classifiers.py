# -*- coding: utf-8 -*-
"""Classifiers"""

import numpy as np
import math
import solver
from implementations import log_1_plus_exp_safe


class LeastSquares:
    """Least squares classifier"""

    def __init__(self, verbose=False, max_evaluations=100):
        """
        Constructor

        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """

        self.verbose = verbose
        self.max_evaluations = max_evaluations

    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # find weights
        self.w = np.linalg.solve(X.T @ X, X.T @ y)


    def function_object(self, w, y, X):
        """
        Function Object.

        :param y: answers
        :param X: data
        :param w: weights
        :return: loss, gradient
        """

        # dimensions
        n, d = X.shape

        # compute error
        e = y - X @ w

        # compute loss
        f = 1/(2 * n) * np.sum(e ** 2)

        # compute gradient
        g = - 1 / n * X.T @ e

        return f, g

    def predict(self, X):
        """
        Predict

        :param X: data
        :return: answer prediction
        """

        return np.sign(X @ self.w)

    
class LeastSquaresL2(LeastSquares):
    """L2-regularized Least Squares"""
    
    def __init__(self, lambda_, verbose=False, max_evaluations=100):
        """
        Constructor

        :param lambda: regularization strength
        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """
        
        self.lambda_ = lambda_
        super().__init__(verbose, max_evaluations)
    
    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # find weights
        self.w = np.linalg.solve(X.T @ X + n * self.lambda_ * np.eye(d), X.T @ y)
        
    def function_object(self, w, y, X):
        """
        Function Object.

        :param y: answers
        :param X: data
        :param w: weights
        :return: loss, gradient
        """
        
        f, g = super().function_object(w, y, X)

        # add regularization
        f += self.lambda_ / 2 * w.dot(w)
        g += self.lambda_ * w

        return f, g
    
    
class LeastSquaresL1(LeastSquares):
    """L1-regularized Least Squares"""
    
    def __init__(self, lambda_, verbose=False, max_evaluations=100):
        """
        Constructor

        :param lambda: regularization strength
        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """
        
        self.lambda_ = lambda_
        super().__init__(verbose, max_evaluations)
    
    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # fit weights
        self.w, f = solver.gradient_descent_L1(self.function_object, self.w, self.lambda_,
                                               self.max_evaluations, y, X, verbose=self.verbose)
    

    
class LogisticRegression:
    """Logistic Regression"""

    def __init__(self, verbose=False, max_evaluations=100):
        """
        Constructor

        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """

        self.verbose = verbose
        self.max_evaluations = max_evaluations

    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # fit weights
        self.w, f = solver.gradient_descent(self.function_object, self.w, 
                                            self.max_evaluations, y, X, verbose=self.verbose)

    def function_object(self, w, y, X):
        """
        Function Object.

        :param y: answers
        :param X: data
        :param w: weights
        :return: loss, gradient
        """

        pred = y * X.dot(w)

        # function value
        f = np.sum(log_1_plus_exp_safe(-pred))

        # gradient value
        res = - y / (1. + np.exp(pred))
        g = X.T.dot(res)

        return f, g

    def predict(self, X):
        """
        Predict

        :param X: data
        :return: answer prediction
        """

        return np.sign(X @ self.w)

    
class LogisticRegressionL2(LogisticRegression):
    """L2-regularized Logistic Regression"""

    def __init__(self, lambda_=1.0, verbose=False, max_evaluations=100):
        """
        Constructor

        :param lambda_: lambda of L1 regularization
        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """
        
        self.verbose = verbose
        self.max_evaluations = max_evaluations
        self.lambda_ = lambda_

    def funObj(self, w, y, X):
        """
        Function Object

        :param w: weight
        :param y: answers
        :param X: data
        :return: loss, gradient
        """
        # Obtain normal loss and gradient using the superclass
        f, g = super().funObj(w, y, X)

        # Add L2 regularization
        f += self.lambda_ / 2. * w.dot(w)
        g += self.lambda_ * w

        return f, g

    
class LogisticRegressionL1(LogisticRegression):
    """L1-regularized Logistic Regression"""

    def __init__(self, lambda_=1.0, verbose=False, max_evaluations=100):
        """
        Constructor

        :param lambda_: lambda of L2 regularization
        :param verbose: print out information
        :param max_evaluations: maximum number of evaluations
        """
        
        self.verbose = verbose
        self.max_evaluations = max_evaluations
        self.lambda_ = lambda_
       
        
    def fit(self, y, X):
        """
        Finds weights to fit the data to the model

        :param y: answers
        :param X: data
        """

        # dimensions
        n, d = X.shape

        # initial weight vector
        self.w = np.zeros(d)

        # fit weights
        self.w, f = solver.gradient_descent_L1(self.function_object, self.w, self.lambda_,
                                               self.max_evaluations, y, X, verbose=self.verbose)
    
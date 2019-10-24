# -*- coding: utf-8 -*-
"""FeatureEngineering"""

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



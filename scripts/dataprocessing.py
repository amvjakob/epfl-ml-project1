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

    if mean is None:
        mean = np.mean(x, axis=0)

    if std is None:
        std = np.std(x, axis=0)

    return (x - mean) / std, mean, std


def remove_NaN_features(x, threshold=0.0):
    """
    Removes the feature if it has more than a certain percentage of -999 values.

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
    Replaces the -999 values of x by the mean of that feature vector.

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


def remove_features(data, features, feats):
    """
    This function removes features from the data and the features list.

    :param data: tX data
    :param features: list of all features from load_csv
    :param feats: array of strings containing the features we want to remove
    :return: new data, new features
    """

    idx_to_remove = -1 * np.ones(len(feats))
    removed = []

    for i, feat in enumerate(feats):
        if feat in features:
            idx_to_remove[i] = features.index(feat)
            removed.append(feat)

    return np.delete(data, idx_to_remove, 1), np.delete(features, idx_to_remove)


def binarize_undefined(data, features, feats):
    """
    Additive Binarization of NaNs in a database.

    Adds a feature whose value is 1 if the value is defined in wanted feature
    column (and 0 otherwise).

    :param data: data with NaN values
    :param feats: features to take into account for additive binarization
    :return: new data
    """

    done = []

    for i, feat in enumerate(feats):

        # check if wanted feature is in feature list
        if feat in features:
            # find index of feature to analyze
            idx_to_analyze = features.index(feat)

            # expand data with 1 where value is defined, 0 where value is undefined
            data = np.c_[data, data[:, 2 * (idx_to_analyze != -999) - 1]]

            # add feature name
            features.append(features[idx_to_analyze] + "_NAN_BINARIZED")
            done.append(feat)

    return data, features
            
            
def cross_validate(y, tx, classifier, ratio, n_iter):
    """
    Cross-validate classifier.

    Shuffles dataset randomly, divides tx in train and
    test based on ration to compute accuracy, repeats procedure n_iter times.

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


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold cross-validation."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validate_kfold(y, x, classifier, k_fold):
    """
    K-fold cross-validation of a classifier.

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


def compute_accuracy(ypred, yreal):
    """
    Compute accuracy of prediction.

    :param ypred: predicted y
    :param yreal: real y
    :return: mean accuracy
    """

    return np.mean(ypred == yreal)


def split_data(features, X, y=None):
    """
    Splits the data matrix X into partitions based on the integer feature 'PRI_jet_num'
    
    :param features: feature names
    :param X: examples
    :param y: labels (optional)
    :return indices of split for every subset, X_split, y_split
    """
    # features that are undefined for some subsets
    undef_feature_for = {
        'DER_deltaeta_jet_jet'   : [0, 1],
        'DER_mass_jet_jet'       : [0, 1],
        'DER_prodeta_jet_jet'    : [0, 1],
        'DER_lep_eta_centrality' : [0, 1],
        'PRI_jet_num'            : [0, 1, 2, 3],
        'PRI_jet_leading_pt'     : [0],
        'PRI_jet_leading_eta'    : [0],
        'PRI_jet_leading_phi'    : [0],
        'PRI_jet_subleading_pt'  : [0, 1],
        'PRI_jet_subleading_eta' : [0, 1],
        'PRI_jet_subleading_phi' : [0, 1],
        'PRI_jet_all_pt'         : [0]
    }

    # the feature based on which we split X
    jet_num_feature = "PRI_jet_num"
    jet_levels = 4

    # build valid features for every subset of X
    features_split = []
    for jet in range(jet_levels):
        valid_features = [ f for f in features if not ((f in undef_feature_for) and (jet in undef_feature_for[f])) ]
        features_split.append(valid_features)
        
    # split data based on jet level (vertical split)
    X_ = X.copy()
    
    split_indices = [
        X_[:,features.index(jet_num_feature)] == i for i in range(jet_levels)
    ]
    X_split = [
        X_[X_[:,features.index(jet_num_feature)] == i,:] for i in range(jet_levels)
    ]
    if y is None:
        y_split = None
    else:
        y_split = [
            y[X_[:,features.index(jet_num_feature)] == i] for i in range(jet_levels)
        ]

    # only keep relevant features (horizontal split)
    for i, x in enumerate(X_split):
        indices = [ features.index(feature) for feature in features_split[i] ]
        indices_bool = [ e in indices for e in range(len(features)) ]
        X_split[i] = x[:,indices_bool]
        
    return split_indices, X_split, y_split


def build_poly_no_interaction(X, degree):
    """
    Build a polynomial expansion of X without interaction terms.
    
    :param X: data
    :param degree: degree of expansion
    """
    result = X.copy()
    for d in range(2, degree+1):
        # np.power() behaves strangely sometimes, so we do the multiplication manually
        power = X.copy()
        for i in range(d - 1):
            power = power * X
            
        result = np.hstack((result, power))
        
    return result

def build_X(X, d_int, d_sq):
    """
    Expand X with integer and/or half-powers.
    
    :param X: examples
    :param d_int: degree of integer powers
    :param d_sq: ceil of degree of half-powers (expansion will be up to d_sq - 0.5)
    """    
    X_ = X.copy()
    
    ints = []
    sqrts = []
    
    # build integer powers
    if d_int > 0:
        ints = build_poly_no_interaction(X_, d_int)
      
    # build half-powers (0.5, 1.5, 2.5, etc.)
    if d_sq > 0:
        sqrts = np.sqrt(np.abs(X_))
        if d_sq > 1:
            width = sqrts.shape[1]
            int_power = np.abs(build_poly_no_interaction(X_, d_sq - 1))
            
            half_power = sqrts.copy()
            for i in range(d_sq - 1):
                # add half power i - 0.5
                half_power = np.hstack((half_power, sqrts * int_power[:,(width*i):(width*(i+1))]))
                
            sqrts = np.hstack((sqrts, half_power))
    else:
        return ints

    # concat
    X_ = np.hstack((ints, sqrts))
    return X_

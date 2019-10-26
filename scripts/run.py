# -*- coding: utf-8 -*-
"""Run"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from dataprocessing import *
from classifiers import *
from solver import *


"""
TRAIN DATA PULL
"""

# fetch train data
DATA_TRAIN_PATH = "../data/train.csv"
y, tX, ids, features = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

"""
FEATURE ENGINEERING
"""

# split data
indices_split, X_split, y_split = split_data(features, tX, y)

# standardize data
X_split_std, mean_split, std_split = [], [], []
for X in X_split: 
    # remove features with more than 20% of NaN and standardize
    X_std, mean_std, std_std = standardize(remove_NaN_features(X, 0.2))
    
    X_split_std.append(X_std)
    mean_split.append(mean_std)
    std_split.append(std_std)

# model values
best_lambda = 3.5938136638046255e-12
best_deg_int = 10
best_deg_sq = 5

def model_split_data(X):
    return build_X(X, best_deg_int, best_deg_sq)

# build expanded data
X_split_poly = [ model_split_data(X) for X in X_split_std ]


"""
FIT AND PREDICT
"""

# train actual models
models = []
for X_, y_ in zip(X_split_poly, y_split):
    lse = LeastSquaresL2(best_lambda)
    lse.fit(y_, X_)
    models.append(lse)

    
"""
TEST DATA PULL
"""

# fetch test data
DATA_TEST_PATH = "../data/test.csv"
y_test, tX_test, ids_test, features_test = load_csv_data(DATA_TEST_PATH, sub_sample=False)

# split
test_split_indices, X_test_split, _ = split_data(features_test, tX_test)

# standardize
X_test_split_std = []
for X, mean, std in zip(X_test_split, mean_split, std_split): 
    # remove features with more than 20% of NaN and standardize
    X_test_std, _, _ = standardize(remove_NaN_features(X, 0.2), mean, std)
    
    X_test_split_std.append(X_test_std)
    
# expand
X_test_split_poly = [ model_split_data(X) for X in X_test_split_std ]

# predictions using new model
y_pred = np.ones(tX_test.shape[0])
for model, X, indices in zip(models, X_test_split_poly, test_split_indices):
    y_pred[indices] = model.predict(X)


"""
OUTPUT PREDICTIONS
"""

OUTPUT_PATH = "../results/predictions.csv"
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

# The Higgs boson Machine Learning challenge

The Higgs boson is an elementary particle which explains why other particles have a mass. Measurements stemming from high-speed collisions of protons at the European Organization for Nuclear Research (CERN) were made available with the aim of predicting whether the collision by-products are an actual boson. 

A first part dedicated to descriptive statistics allows a broader overview on the dataset structure and particularities. 

A preprocessing step can include different combinations of the following methods : (1) splitting the dataset on the feature PRI-jet-num, (2) replacing undefined datapoints by the median/mean or binerize them, (3) performing a polynomial expansion, (4) standardizing and eventually (5) handling outliers. 

A Least Squares and Logistic Regressions are subsequently implemented and legitimized by means of a 5 folds cross validation. 


## Getting Started

No external libraries are necessary besides Numpy and Matplolib for visualisation. 


## Description


### `run.py`

Running that file allows producing exactly the same .csv predictions that were used in the best submission into the competition platform. It is self contained and only requires access to the data and files described below. 

---
### `classifier.py`

Modular code which enables an easy use of its classes. To both of the models of interest, Least Squares and Logistic Regression, an L1 or L2 regularization can be applied. 

---
### `solver.py`

The models above mentionned are either resolved directly or thanks to a Gradient Descent/Subgradient Descent. The file contains two functions, *gradient_descent* and *gradient_descent_L1* that are necessary within the `classifier.py` file. 

---
### `dataprocessing.py`

Includes all the functions needed for the feature engineering and further steps such as cross-validation and optimisation. 

* *standardize*
* *remove_NaN_features, remove_features*
* *replace_NaN_by_mean/replace_NaN_by_median*
* *binarize_undefined*

* *compute_accuracy, cross_validate, cross_validate_kfold*
* *find_max_hyperparam*
* *log_1_plus_exp_safe*

---
### `implementation.py`

List of functions that had to be implemented in the framework of the project. For the sake of completeness, the solvers above mentionned are encoded here as well. 

* *least_squares, least_squaresGD, batch_iter,  least_squaresSGD*
* *ridge_regression, lasso_regression*
* *logistic_regression, reg_logistic_regression, logistic_regression_sparse*
* *GD, GD_linesearch, GD_L1*

---
### `proj1_helpers.py`

*load_csv_data*, *predict_labels* and *create_csv_submission* that were given as helpers. 

---

### `plotting.ipynb`

A call to its function will let the user obtain a visual comparison of the accuracies obtained with different combination of data preprocessing steps and models applied. 

---
### `project1.ipynb`

That jupyter notebook reflects the thought process. It starts with splitting the dataset on the feature PRI-jet-num. The different models are then tested and optimized. 

---
### `Dataset_visualisation.ipynb`

Dedicated to descriptive statistics, its output is a figure standing for a visualisation of the features correlations. 



## Authors

* Arnaud DHAENE 
* Anthony JAKOB
* Maëlle ROMERO GRASS


## Acknowledgments

CF RAPPORT
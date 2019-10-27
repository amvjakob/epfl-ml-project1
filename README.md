# The Higgs boson Machine Learning challenge

The Higgs boson is an elementary particle which explains why other particles have a mass. Measurements stemming from high-speed collisions of protons at the European Organization for Nuclear Research (CERN) were made available with the aim of predicting whether the collision by-products are an actual boson or background noise. 

A first part dedicated to descriptive statistics allows a broader overview of the dataset structure and particularities. 

A preprocessing step can include different combinations of the following methods: (1) splitting the dataset on the feature PRI-jet-num, (2) replacing undefined datapoints by the median/mean or binarize them, (3) performing a polynomial expansion, (4) standardizing and eventually (5) handling outliers. 

A Least Squares and Logistic Regressions are subsequently implemented and legitimized by means of a 5-fold cross validation. 


## Getting Started

No external libraries are necessary besides Numpy and Matplolib for visualisation. 


## Codes description


### `run.py`

Running that file produces exactly the same .csv predictions that were used in the best submission on the competition platform. It is self-contained and only requires access to the data and files described below.

---
### `implementations.py`

List of functions that had to be implemented in the framework of the project. For the sake of completeness, the solvers above mentionned are coded here as well. 

* *least_squaresGD, least_squaresSGD, least_squares, ridge_regression*
* *logistic_regression, reg_logistic_regression*
* *GD, log_1_plus_exp_safe*

---
### `dataprocessing.py`

Includes all the functions needed for the feature engineering and further steps such as cross-validation and optimisation. 

* *standardize*
* *remove_NaN_features, remove_features*
* *replace_NaN_by_mean/replace_NaN_by_median*
* *binarize_undefined*

* *cross_validate, cross_validate_kfold, compute_accuracy*
* *split_data, build_X*

---
### `classifier.py`

Useful classes for Least Squares and Logistic Regression, to which either L1 or L2 regularization can be applied.

---
### `solver.py`

The models in `classifier.py` are either resolved directly or with gradient/subgradient descent. The file contains two functions, *gradient_descent* and *gradient_descent_L1* that are used within the `classifier.py` file.

---
### `proj1_helpers.py`

Functions *load_csv_data*, *predict_labels* and *create_csv_submission* that were given as helpers. 

---

### `plotting.py`

The function *boxplot_models* allows a visual comparison of the accuracies obtained with different combination of data preprocessing steps and models applied. A second function *surface3d_model* is dedicated to the optimization steps. It depicts cross-validated accuracies as a function of the polynomial expansion degree and the L2 regularization hyperparameter. 

---
### `project1.ipynb`

The notebook starts with splitting the dataset on feature PRI-jet-num. After standardizing, Least Squares and Logistic Regressions are tested, both being as well L1 and L2 regularized. A Grid Search is performed on the polynomial expansion degree and the regularization hyperparameter. The final model can then be trained and the submission file produced. 

---
### `myrun.ipynb`

That jupyter notebook reflects the score improvement steps, which are :
* (A) Base
* (B) Base with offset and standardisation
* (C) Base with offset, standardisation, and NaN to median
* (D) Base with offset, standardisation, additive binarization, removal of all NaN features except for *DER-mass-mmc*
* (E) Base with offset, standardisation, NaN to median, and outliers capped to 5% and 95% percentiles 
* (F) Data split in function of *PRI-jet-num*, and L2 regularized with $\lambda = 10^{-4}$ 
* (G) Data split in function of *PRI-jet-num*, polynomial expansion using d<sub>int</sub>= 2, d<sub>sqrt</sub>= 1 and L2 regularized with $\lambda = 10^{-4}$
* (H) Data split in function of *PRI-jet-num*, polynomial expansion using d<sub>int</sub>=11, d<sub>sqrt</sub>= 3, and L2 regularized with $\lambda = 1.46 \cdot 10^{-8}$

---
### `visualisation.ipynb`

Dedicated to descriptive statistics, its output is a figure standing for a visualisation of the features' correlations. 



## Authors

* Arnaud DHAENE 
* Anthony JAKOB
* Maëlle ROMERO GRASS


## Acknowledgments

We would like to acknowledge the use and adaption of code provided my Mark Schmidt during the machine learning courses CPSC 340 and CPSC 540 at UBC, Vancouver.
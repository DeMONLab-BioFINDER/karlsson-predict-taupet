from itertools import product
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.feature_selection import SelectPercentile,SelectFdr,f_regression,mutual_info_regression
from sklearn.model_selection import KFold



def build_namestring(options, sep='.'):
    """
    Builds a namestring separated by "." for all combinations in options. 

    Inputs:
        - options (list): List of lists with combination keywords.
        - sep (String): option separator in string.
        
    Outputs:
        - out_strings (list): List of all possible combinations in single strings separated by sep.
    """
    iterations = list(product(*options))
    out_strings = []
    for it in iterations:
        string = ''
        for i,item in enumerate(it):
            if i == (len(it)-1):
                string+=item
            else:
                s = '%s' + sep
                string+=s%item
        out_strings.append(string)
    return out_strings



def get_multiple_cv(sids):
    """
    Creates a 10-fold cross-validation from a series of non-unique sids where a unique sid is always placed in the same split (either train or validation index).

    Inputs:
        - sids (series): Series of sid names with overlap.

    Outputs: 
        - cv (list): list of cross-validation indices for 10 folds. 
    """
    unique_sids = sids.unique()
    kf = KFold(n_splits=10,shuffle=True)
    cv = []
    for i, (train_index, val_index) in enumerate(kf.split(unique_sids)):
        idx_train = [idx for i,idx in zip(sids,sids.index) if i in unique_sids[train_index]]
        idx_val = [idx for i,idx in zip(sids,sids.index) if i in unique_sids[val_index]]
        cv.append((idx_train,idx_val))
    return cv



def get_dicts():
    """
    Returns dictionaries corresponding to pipeline steps.

    Outputs: 
        - dim_dict (Dictionary): Dictionary of dimensionality reduction steps.
        - estim_dict (Dictionary): Dictionary of estimators.
        - params_dict (Dictionary): Dictionary of search spaces for each estimator.
        - n_iter_dict (Dictionary): Dictionary of number of iterations in Bayesian Optimization for each estimator.
    """
    dim_dict = {'f_perc': SelectPercentile(f_regression,percentile=10),
               'mir_perc': SelectPercentile(mutual_info_regression,percentile=10),
               'f_perc50': SelectPercentile(f_regression,percentile=50),
               'mir_perc50': SelectPercentile(mutual_info_regression,percentile=50) } #'f_FDR': SelectFdr(f_regression),


    estim_dict = {'lin': LinearRegression(),
                 'elnet': ElasticNet(),
                 'SVR': SVR(),
                 'RF': RandomForestRegressor(),
                 'KNN': KNeighborsRegressor(),
                 'XGB': xgb.XGBRegressor(),
                 'CatB': CatBoostRegressor()}

    params_dict = {'lin': {'estimator__fit_intercept':[True],
                          'estimator__n_jobs': [1]},
                 'elnet': {'estimator__alpha':Real(1e-8,1),
                          'estimator__l1_ratio':Real(1e-8,1)},
                 'SVR': {'estimator__kernel': Categorical(['poly', 'rbf']), 
                       'estimator__C': Real(1e-5, 1e3,prior='log-uniform')},
                 'RF': {'estimator__max_depth': Integer(2,12),
                    'estimator__n_estimators': Integer(5,500),
                    'estimator__n_jobs': [2]},
                 'KNN': {'estimator__n_neighbors': Integer(3,26),
                        'estimator__n_jobs': [1]},
                 'XGB': {'estimator__max_depth': Integer(2,12),
                      'estimator__eta': Real(0.01, 0.3,prior='log-uniform'),
                      'estimator__reg_lambda': Real(1e-8, 1.0, prior='log-uniform'),
                      'estimator__reg_alpha': Real(1e-8, 1.0,prior='log-uniform'),
                      'estimator__n_estimators': Integer(5,500),
                      'estimator__n_jobs': [1]},
                 'CatB': {'estimator__depth': Integer(2,12),
                       'estimator__n_estimators': Integer(5,500),
                       'estimator__learning_rate': Real(0.01,0.3, prior='log-uniform'),
                       'estimator__verbose': [False],
                       'estimator__thread_count': [3]}}

    n_iter_dict = {'lin': 1,
                 'elnet': 20,
                 'SVR': 30,
                 'RF': 50,
                 'KNN': 5,
                 'XGB': 50,
                 'CatB': 40}
    return dim_dict, estim_dict, params_dict, n_iter_dict

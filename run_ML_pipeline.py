pimport pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import fns as fns
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys
import os
os.environ['OMP_NUM_THREADS'] = "1"

warnings.simplefilter("ignore", UserWarning)


@ignore_warnings(category=ConvergenceWarning)
def run_ML_pipeline(df, pipeline_combination, outcome='tnic_cho_com_I_IV',id_name='sid',visit_name='Visit',sep='.'):
    """
    Runs a ML pipeline for a dataframe containing the outcome variable "outcome". The imputation, inclusion type, dimensionality reduction and estimator of the pipeline is specified as a string "pipeline_combination" with "." as separator. For parameter tuning, Baysian optimization is performed. The evaluation is performed as a 10-fold-cross-validation with metric negative mean absolute percentage error. 

    Inputs:
        - df (DataFrame): Dataframe containing columns outcome, sid and features.
        - pipeline_combination (String): Pipeline specification with "." as separator.
        - outcome (String): Name of outcome variable in df.
        - id_name (String): Name of subject ID variable in df. Should be unique for every unique subject.
        - visit_name (String): Name of visit variable in df. Should be 0 for baseline visit.
        - sep (String): option separator in string.

    Outputs:
        - best_score (float): best negative mean absolute percentage error after Baysian Optimization on the hold out data. 
        - best_params (dict): Parameter setting that gave the best results on the hold out data.
    """
    

    df = df.sample(frac=1).reset_index(drop=True)
    
    # Extract information in pipeline combination
    [imputation, incl_type, dim_red, estimator] = pipeline_combination.split(sep)

    # Initialize list of all steps in pipeline
    steps = []

    # Add KNN imputation to pipline or dropna
    if imputation == 'KNNimp':
        steps.append(('imputer', KNNImputer()))
        save = df[outcome].dropna().index
        df = df.loc[save].reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    
    # Add standardscaler to pipeline
    steps.append(('scaler', StandardScaler()))
    
    # Create cv for inclusion types
    if incl_type == 'baseline':
        df = df[df[visit_name] == 0].reset_index(drop=True)
        cv = 10
    elif incl_type == 'multiple':
        cv =  fns.get_multiple_cv(df[id_name])
    else:
        raise ValueError('Inclusion type must be set to "baseline" or "multiple"')
    
    # Extract X, y and get dictionaries for pipeline instructions
    y = df[outcome]
    X = df.drop(columns = [outcome,visit_name]) # ADD id_name if multiple
    dim_dict, estim_dict, params_dict, n_iter_dict = fns.get_dicts()
        
    # Add dimensionality reduction to pipeline (if not None)
    if dim_red == 'RF_imp':
        clf = RandomForestRegressor(n_estimators=30)
        idx_rf = X.dropna().index
        clf.fit(X.loc[idx_rf],y.loc[idx_rf])
        vec = sorted(clf.feature_importances_,reverse=True)
        s = np.cumsum(vec) < 0.95
        idx_threshold = len(s[s])
        threshold = vec[idx_threshold]
        steps.append(('dimred', SelectFromModel(clf, prefit=False,threshold=threshold-0.00001, importance_getter = 'feature_importances_')))  #max_features=int(np.ceil(len(X.columns) * 0.1))
    elif dim_red == 'RFE':
        clf = RandomForestRegressor(n_estimators=30)
        idx_rf = X.dropna().index
        clf.fit(X.loc[idx_rf],y.loc[idx_rf])
        steps.append(('dimred', RFE(clf,n_features_to_select=max(3,round(len(X.columns)*0.1)),step=0.2,verbose=False)))
    elif dim_red != 'None':
        steps.append(('dimred', dim_dict[dim_red]))

    # Add estimator to pipline
    steps.append(('estimator', estim_dict[estimator])) 

    # Create pipeline
    pipe = Pipeline(steps)

    # Perform cross-validation and Baysian optimzation of hyperparameters, and return best results
    opt = BayesSearchCV(pipe, search_spaces = params_dict[estimator], cv = cv, n_jobs = 1, n_iter=n_iter_dict[estimator], scoring = 'neg_mean_absolute_percentage_error')
    opt.fit(X, y)
    best_score = opt.best_score_
    best_params = opt.best_params_

    return best_score, best_params

    
    

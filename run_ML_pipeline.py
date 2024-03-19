import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_predict,cross_val_score,cross_validate
import fns as fns
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys
import os
from sklearn.base import clone
os.environ['OMP_NUM_THREADS'] = "1"
warnings.simplefilter("ignore", UserWarning)
from collections import OrderedDict


@ignore_warnings(category=ConvergenceWarning)
def bays_opt_ML_pipeline(df, pipeline_combination, classifier=False, outcome='tnic_cho_com_I_IV', id_name='sid', visit_name='Visit', sep='.', scoring='neg_mean_squared_error'):
    """
    Optimizes a ML classification or regression pipeline for a dataframe containing the outcome variable "outcome". The imputation, inclusion type, dimensionality reduction and estimator of the pipeline is specified as a string "pipeline_combination" with "sep" as separator. For parameter tuning, Bayesian optimization is performed. The evaluation is performed as a 10-fold-cross-validation optimizing hyperparameters on "scoring". 

    Inputs:
        - df (DataFrame): Dataframe containing columns outcome and features.
        - pipeline_combination (String): Pipeline specification with "." as separator.
        - classifier (boolean): regression (False) or classification (True).
        - outcome (String): Name of outcome variable in df.
        - id_name (String): Name of subject ID variable in df. Should be unique for every unique subject.
        - visit_name (String): Name of visit variable in df. Should be 0 for baseline visit.
        - sep (String): option separator in string.
        - scoring (String): metric to optimize hyperparameters on.

    Outputs:
        - best_score (float): best score after Baysian Optimization on the held out cv data. 
        - best_params (dict): Parameter settings that gave the best results on the held out cv data.
    """
    # Extract pipeline steps from pipeline combination
    [imputation, incl_type, dim_red, estimator] = pipeline_combination.split(sep)
    
    # Build pipeline 
    pipe, estimator, cv, X, y = build_ML_pipeline(df, imputation, incl_type, dim_red, estimator, classifier=classifier, outcome=outcome, id_name=id_name, visit_name=visit_name)
    
    # Get Bayesian optimzation parameters and parameter intervals
    params_dict, n_iter_dict = fns.get_bays_opt(classifier)
    
    # Perform cross-validation and Baysian optimzation of hyperparameters, and return best results
    opt = BayesSearchCV(pipe, search_spaces = params_dict[estimator], cv = cv, n_jobs = 1, n_iter=n_iter_dict[estimator], scoring = scoring)
    opt.fit(X, y)
    best_score = opt.best_score_
    best_params = opt.best_params_

    return best_score, best_params


def get_best_CV_prediction(all_input,df,classifier=False, outcome = 'tnic_cho_com_I_IV',id_name='sid', visit_name='Visit'):
    """
    Evaluates a ML classification or regression pipeline for a dataframe containing the outcome variable "outcome". The imputation, inclusion type, dimensionality reduction and estimator of the pipeline is specified from previous optimization results in "all_input". The evaluation is performed as a 10-fold-cross-validation. 

    Inputs:
        - all_input (DataFrame): Dataframe with results from testing different ML pipelines (using e.g. bays_opt_ML_pipeline)
        - df (DataFrame): Dataframe containing columns outcome and features.
        - classifier (boolean): regression (False) or classification (True).
        - outcome (String): Name of outcome variable in df.
        - id_name (String): Name of subject ID variable in df. Should be unique for every unique subject.
        - visit_name (String): Name of visit variable in df. Should be 0 for baseline visit.

    Outputs:
        - y (list): List of true y.
        - y_pred_input: List of predicted y in cross-validation.
        - (y_pred_input_proba): List of predicted probabilites if classification problem.
    """
    # Selects the pipeline with HIGHEST score in all_input
    sorted_all_inputs = all_input.sort_values(by='Best_score').reset_index(drop=True)
    best = sorted_all_inputs.iloc[0]
    
    # Extract the pipeline specifications of the best pipeline
    imputation = best['Imputation']
    incl_type = best['Inclusion_type']
    dim_red = best['Dim_Red']
    estimator = best['Estimator']
    
    # Build pipeline from specifications
    pipe, estimator, cv, X, y = build_ML_pipeline(df,imputation, incl_type, dim_red, estimator, classifier=classifier, outcome=outcome, id_name=id_name, visit_name=visit_name)
    params = dict(eval(best['Best_params']))
    pipe.set_params(**params)
    pipe1 = clone(pipe)
    
    # Perform cross-validated prediction
    y_pred_input = cross_val_predict(pipe, X, y, cv = cv)
    if classifier:
        y_pred_input_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')
        return y, y_pred_input, y_pred_input_proba,pipe1
    else:
        return y, y_pred_input,pipe1



def build_ML_pipeline(df, imputation, incl_type, dim_red, estimator, classifier=False, outcome='tnic_cho_com_I_IV', id_name='sid', visit_name='Visit'):
    """
    Builds an ML pipeline based on specification of imputation, inclusion type, dimensionality reduction and estimator. 
    
    Inputs:
        - df (DataFrame): Dataframe containing columns outcome, sid and features.
        - imputation (String): imputation type (KNN or None)
        - incl_type (String): inclusion type (baseline or multiple if longitudinal data exist)
        - dim_red (String): dimensionality reduction type ('None','RF_imp','f_perc','f_perc50', 'mir_perc','mir_perc50','RFE' or 'RFE50')
        - estimator (String): estimator type ('lin','elnet', 'SVR', 'RF', 'KNN', 'XGB' or 'CatB')
        - classifier (boolean): regression (False) or classification (True).
        - outcome (String): Name of outcome variable in df.
        - id_name (String): Name of subject ID variable in df. Should be unique for every unique subject.
        - visit_name (String): Name of visit variable in df. Should be 0 for baseline visit.

    Outputs:
        - pipe (Pipeline): ML pipeline.
        - estimator (String): estimator in pipeline.
        - cv (list or int): list of cross-validation indices for 10 folds or number of folds.
        - X (DataFrame): DataFrame with features.
        - y (DataFrame): DataFrame with outcome.
    """
    
    # Shuffle order of data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Initialize list of all steps in pipeline
    steps = []
    
    # Add KNN imputation to pipline or dropna
    if imputation == 'KNNimp':
        steps.append(('imputer', KNNImputer()))
        save = df[outcome].dropna().index
        df = df.loc[save].reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    
    # Create cv for inclusion types
    if incl_type == 'baseline':
        df = df[df[visit_name] == 0].reset_index(drop=True)
        cv = 10
    elif incl_type == 'multiple':
        cv =  fns.get_multiple_cv(df[id_name])
    else:
        raise ValueError('Inclusion type must be set to "baseline" or "multiple"')
    
    # Add standardscaler to pipeline
    steps.append(('scaler', StandardScaler()))
    
    X = df.drop(columns = [outcome, visit_name, id_name])
    y = df[outcome]
    if classifier:
        dim_dict, estim_dict = fns.get_dicts_classification()
    else:
        dim_dict, estim_dict = fns.get_dicts_regression()
        
    # Add dimensionality reduction to pipeline
    if dim_red != 'None':
        steps.append(('dimred', add_dim_red(dim_red,dim_dict,X,y,classifier)))

    # Add estimator to pipline
    steps.append(('estimator', estim_dict[estimator])) 

    # Create pipeline
    pipe = Pipeline(steps)
    
    return pipe, estimator, cv, X, y
    


def add_dim_red(dim_red,dim_dict,X,y,classifier):
    """
    Returns a dimensionality reduction based on given specifications.
    
    Inputs:
        - dim_red (String): dimensionality reduction type ('None','RF_imp','f_perc','f_perc50', 'mir_perc','mir_perc50','RFE' or 'RFE50')
        - dim_dict (Dictionary): Dictionary of dimensionality reduction steps.
        - X (DataFrame): DataFrame with features.
        - y (DataFrame): DataFrame with outcome.
        - classifier (boolean): regression (False) or classification (True).

    Outputs:
        - dimensionality reduction method.
    """

    # RF imp: use features making up 95% of the cumulative feature importance in a RF classifier/regressor
    if dim_red == 'RF_imp':
        if classifier:
            clf = RandomForestClassifier(n_estimators=30)
        else:
            clf = RandomForestRegressor(n_estimators=30)
        idx_rf = X.dropna().index
        clf.fit(X.loc[idx_rf],y.loc[idx_rf])
        vec = sorted(clf.feature_importances_,reverse=True)
        s = np.cumsum(vec) < 0.95
        idx_threshold = len(s[s])
        threshold = vec[idx_threshold]
        
        return SelectFromModel(clf, prefit=False,threshold=threshold-0.00001, importance_getter = 'feature_importances_') 
    
    # RFE: use "lim" proportion of features based on recursive feature elimination from a RF classifier/regressor
    elif (dim_red == 'RFE') | (dim_red == 'RFE50'):
        lim = 0.5 if dim_red == 'RFE50' else 0.1
        if classifier:
            clf = RandomForestClassifier(n_estimators=30)
        else:
            clf = RandomForestRegressor(n_estimators=30)
        clf.fit(X,y)
        return RFE(clf,n_features_to_select=max(1,round(len(X.columns)*lim)),step=0.2,verbose=False)
    
    # use one of the other methods defined in the dimensionality reduction dictionary
    return dim_dict[dim_red]
    

    

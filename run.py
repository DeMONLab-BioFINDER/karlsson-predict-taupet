import pandas as pd
import fns as fns
import run_ML_pipeline as rmlp
import time
from multiprocessing import Pool
import pickle
import os
from pathlib import PurePath
path = PurePath(os.getcwd())
import sys
import numpy as np


## Specify save name
save_name = 'run1' ### INPUT SAVE NAME

### Import harmonized data
df = pd.read_csv('simulated_data.csv') ### INPUT DATA FILE

###Define variable input blocks
#clinical
clinical_variables = ['age', 'sex','education','mmse_score','adas_delayed_word_recall','apoe_e2', 'apoe_e4']
#plasma
plasma_variables = ['plasma_ptau217','plasma_ptau181','plasma_ptau231','plasma_NTA','plasma_GFAP','plasma_NfL'] 
#MRI
volume = [col for col in df if ((col.startswith('aseg_vol')) or (col.startswith('aparc_grayvol'))) and ('scan' not in col)]
surf_area = [col for col in df if (col.startswith('aparc_surfarea')) and ('scan' not in col)]
ct = [col for col in df if (col.startswith('aparc_ct')) and ('scan' not in col)]
MRI_variables = volume + surf_area + ct + ['samseg_wmhs_WMH_total_mm3','icv_mm3']

###Define outcome
outcome = ['tnic_cho_com_I_IV'] ## EDIT depending on what outcome you want to use

### Specifiy name of visit column and participant id
other_variables = ['Visit','sid']

### Specify what input feature block combination you want to test
variables = clinical_variables + outcome + other_variables ## EDIT accordingly by adding/removing the input feature block of interest. Make sure to always include outcome and other_variables.

### Make sure to always use the same data (remove NaNs)
variables_all = clinical_variables + plasma_variables + MRI_variables + outcome + other_variables 
df_all = df[variables_all].dropna().reset_index(drop=True)

###Create pipeline combinations
options = [['None'],#'KNNimp'], # Imputation
           ['baseline'],#'multiple'], # Inclusion type
           ['None','RF_imp','f_perc','f_perc50', 'mir_perc','mir_perc50','RFE','RFE50'], # Dimensionality Reduction
           ['lin','elnet', 'SVR', 'RF', 'KNN', 'XGB', 'CatB']] # Estimator

sep = '.'
combinations = fns.build_namestring(options,sep=sep)


def task(combination): 
    """
    Creates a task that can be executed by a thread. For a specific pipeline combination, 10-fold-cross validation in with Baysian Optimization of hyperparameters is performed. The best scores and params are saved in a pickle file, together with the pipeline specifications.

    Inputs:
        - combination (String): Pipeline specification with "." as separator.
    """
    df1 = df_all[variables]
    time_bf=time.time()
    np.seterr(invalid='ignore')
    best_score,best_params = rmlp.bays_opt_ML_pipeline(df1.copy(),combination,outcome=outcome[0]) #classifier=True, scoring='accuracy' if classification task
    [imputation, incl_type, dim_red, estimator] = combination.split(sep)
    total_time=int((time.time()-time_bf)/60) 
    result = [best_score, best_params, imputation, incl_type, dim_red, estimator, total_time]
    with open(str(path) + '/'+ combination+save_name+'.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(combination + save_name+' done in ' + str(total_time) + ' minutes.')

    

# Multithreaded execution of all combinations.
before = time.time()       
if __name__ == '__main__':
    
    ## Edit number of processes depending on hardware
    with Pool(processes=8) as pool:
        pool.map(task,combinations) 

    # Gather all results into single df, save as csv and remove all pickle files from single threads.    
    all_results = pd.DataFrame(columns = ['Best_score','Best_params','Imputation','Inclusion_type','Dim_Red','Estimator'])
    names = 'Best_score','Best_params','Imputation','Inclusion_type','Dim_Red','Estimator','Time'
    
    for combination in combinations:
        with open(str(path) +'/'+ combination+save_name+'.pkl', 'rb') as handle: 
            result = pickle.load(handle)
        all_results.loc[len(all_results),names] = result
        os.remove(str(path) + '/' + combination+save_name+'.pkl')
        
    all_results.to_csv('all_results_' + save_name + '.csv')
    after = time.time()
    print("total time: " + str(after-before) + ' seconds.')
    
    


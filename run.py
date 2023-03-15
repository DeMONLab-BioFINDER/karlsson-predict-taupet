import pandas as pd
import fns as fns
import run_ML_pipeline as rmlp
import time
from multiprocessing import Process
import pickle
import os
from pathlib import PurePath
path = PurePath(os.getcwd())

## Specify save name
save_name = 'input1'

### Import harmonized data

df = pd.read_csv("df_train.csv")

#variables to include in df
clinical_variables = ['age', 'gender_baseline_variable','education_level_years_baseline_variable','mmse_score','adas_delayed_word_recall','apoe_genotype_baseline_variable']
plasma_variables =['PL_ptau217_pgml_Lilly_2022','PL_ptau181_pgml_Lilly_2022',
                   'Plasma_ptau231_pgml_UGOT_2023','PL_NTAadjusted_pgmL_Simoa_UGOT_2022',
                   'PL_GFAP_pgmL_Simoa_UGOT_2022','PL_NFlight_pgmL_Simoa_UGOT_2022']
volume = [col for col in df if ((col.startswith('aseg_vol')) or (col.startswith('aparc_grayvol'))) and ('scan' not in col)]
surf_area = [col for col in df if (col.startswith('aparc_surfarea')) and ('scan' not in col)]
ct = [col for col in df if (col.startswith('aparc_ct')) and ('scan' not in col)]
MRI_variables = volume + surf_area + ct + ['samseg_wmhs_WMH_total_mm3','icv_mm3']
output_variable = ['tnic_cho_com_I_IV']
other_variables = ['Visit']


variables_all =  clinical_variables + plasma_variables + MRI_variables + output_variable + other_variables

variables = clinical_variables + output_variable + other_variables

#pipeline combinations
options = [['None'],#'KNNimp'], # Imputation
           ['baseline'],#'multiple'], # Inclusion type
           ['None','RF_imp','f_perc','f_perc50', 'mir_perc','mir_perc50','RFE', 'RFE50'],#'None','RF_imp','f_perc', 'mir_perc','RFE'], # Dimensionality Reduction
           ['lin','elnet', 'SVR', 'RF', 'KNN', 'XGB', 'CatB']] # 'lin','elnet', 'SVR', 'RF', 'KNN', 'XGB', 'CatB']] # Estimator

sep = '.'




df = df[variables_all].dropna()
df = df[variables].reset_index(drop=True)[:30]
combinations = fns.build_namestring(options,sep=sep)



def task(combination): 
    """
    Creates a task that can be executed by a thread. For a specific pipeline combination, 10-fold-cross validation in with Baysian Optimization of hyperparameters is performed. The best scores and params are saved in a pickle file, together with the pipeline specifications.

    Inputs:
        - combination (String): Pipeline specification with "." as separator.
    """
    time_bf=time.time()
    best_score,best_params = rmlp.run_ML_pipeline(df.copy(),combination)
    [imputation, incl_type, dim_red, estimator] = combination.split(sep)
    total_time=int((time.time()-time_bf)/60) 
    result = [best_score, best_params, imputation, incl_type, dim_red, estimator, total_time]
    with open(str(path) + '/'+ combination+save_name+'.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(combination + save_name+' done in ' + str(total_time) + ' minutes.')
    

# Multithreaded execution of all combinations.
before = time.time()       
if __name__ == '__main__':

    processes = [Process(target=task, args=(combination,)) for combination in combinations]
    
    # start the processes
    for process in processes:
        process.start()
        
    # wait for all processes to finish
    for process in processes:
        process.join()
        
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
    


    
    
    
    
    
"""

# FOR LOOP TO COMPARE SPEED WITH MULTITHREADING


names = 'Best_score','Best_params','Imputation','Inclusion_type','Dim_Red','Estimator'
all_results = pd.DataFrame(columns = ['Best_score','Best_params','Imputation','Inclusion_type','Dim_Red','Estimator'])

### for thread X and combination in combinations

before = time.time()
for combination in combinations:
    best_score, best_params = rmlp.run_ML_pipeline(df.copy(),combination)
    [imputation, incl_type, dim_red, estimator] = combination.split('.')
    all_results.loc[len(all_results),names] = [best_score, best_params, imputation, incl_type, dim_red, estimator]
    print(combination + ' done.')
after = time.time()
print("total time: " + str(after-before) + ' seconds.')
all_results.to_csv('all_results.csv')


"""

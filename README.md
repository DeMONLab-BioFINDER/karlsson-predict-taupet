# predict-tau-pet

This repo contains code that was used in a tau-PET machine learning (ML) prediction project, including:
- A rigourous ML pipeline that in a combined grid-search/Baysian optimization setting tests hyperparameter-tunes and evaluates different estimators and feature selection methods for a certain ML regression/classification task.
- Notebooks to plot the results.

A simulated dataframe "simualted_data.csv" is included for easy code testing and a template of how the input data should be formatted.
The ML pipeline can be run by editing only in run.py:
- edit the save name (selected by the user)
- edit the name of the input dataframe (should match the input filename)
- edit the name of the outcome to predict (should match the variable name in the dataframe)
- specify what input feature block combination you want to test (will be a concatenated list of the input feature block lists you want to use)
- if needed, also edit the variable names in the input feature blocks so that they match the ones in the dataframe.

Thereafter, the code is runable directly from the terminal by activating the virtual environment:

```console
source activate "ml-env"
```

and then simply running the command:

```console
python3 run.py
```

Each ML feature selection step + ML estimator combination (currently 56) will be executed on parallell threads, each one saved as a pickle file during the run. When finished, all single pickle files are deleted and results are saved in a csv file called. "all_results_"+save_name+".csv". Running the code on a local MAC (Apple M1 Pro, 16 GB RAM) with the simulated data (100 x 33) took approximately 6 minutes. 

Running the code with a complete dataframe (948 x 247) used in the manuscript on a computer cluster slurm system requesting a full node with 16 CPUS took approximately 8 hours. To facilitate future runs on such a system, environment.yml can be used to create a virtual environment named "ml-env" and start.sh to start a run (sbatch start.sh) with the specifics used in this work.

## Dependencies

  - python=3.9
  - pandas=1.4.4
  - scikit-learn=1.1.2
  - catboost=1.1.1
  - numpy=1.23.3
  - xgboost=1.7.3
  - scikit-optimize=0.9.0

## Overview of .py files

run.py loads the data, creates input block combinations and parallell tasks for each machine learning feature selection and estimator combination. It also saves all combinations as pickle files during the run and finalizes by creating a common csv file for all the results.

run_ML_pipeline.py builds and trains machine learning pipelines in a 10-fold cross-validated setting. 

fns.py contains functions that specifies the hyperparameters to be tuned and how many samples during Bayesian optimization, both for regression and classification tasks. 



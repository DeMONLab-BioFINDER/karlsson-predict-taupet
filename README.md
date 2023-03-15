# tauPETregression


This repo contains code for a tau PET ML project.

## Dependencies

- pandas
- numpy 
- scikit-learn
- scikit-optimize
- xgboost
- catboost

## File Description

fns.py contains general functions. run_ML_pipeline.py contains a function to run different pipeline combinations in a Bayesian optimzation + 10-fold-cross validation setting. run.py contains code for creating multithreaded runs of run_ML_pipeline with each a certain pipeline combination specified for each thread. 

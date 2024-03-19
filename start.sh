#!/bin/bash -l

#SBATCH -A sensXXXXX 
#SBATCH -p core 
#SBATCH -n 16 
#SBATCH -t 30:00:00
#SBATCH -J ml_run 
#SBATCH -C usage_mail

source activate ml-env

python3 run.py

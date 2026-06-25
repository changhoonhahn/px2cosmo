#!/bin/bash
#SBATCH -J nle.mock_z11
#SBATCH -o o/nle.mock_z11.o
#SBATCH -p normal 
#SBATCH -N 1               
#SBATCH -n 1               
#SBATCH --time=00:30:00
#SBATCH -A AST26017


module purge
module load intel


unset PYTHONPATH
source ~/.bashrc 

conda activate jwst


python /home1/11004/chahah/projects/px2cosmo/bin/mock/nde_likelihood.py\
    'q_X_omegasig_z11'\
    --training-data-file '/home1/11004/chahah/work/px2cosmo/mock/mock_N2000_z11.dat'\
    --study-dir '/home1/11004/chahah/work/px2cosmo/mock/nde'\
    --verbose

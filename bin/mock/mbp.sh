#!/bin/sh

#python nde_poisson.py 'q_omegat_log1pN_z11' 'z11' /Users/ch54662/data/px2cosmo/mock/ndes/
#python nde_poisson.py 'q_omegat_log1pN_z9' 'z9' /Users/ch54662/data/px2cosmo/mock/ndes/
#python nde_poisson.py 'q_omegat_log1pN_z7' 'z7' /Users/ch54662/data/px2cosmo/mock/ndes/

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python nde_likelihood.py 'q_X_omegasig_z11' \
    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z11.npy \
    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
    --batch-size 512 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python nde_likelihood.py 'q_X_omegasig_z9' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z9.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 512 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python nde_likelihood.py 'q_X_omegasig_z7' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z7.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 512 --njobs 5 --cpu --verbose

#!/bin/sh
#python mkdata.py 2000 /Users/ch54662/data/px2cosmo/mock/
python mkdata_pop.py 10000 /Users/ch54662/data/px2cosmo/mock/ 100
#python mkdata_poisson.py 20000 /Users/ch54662/data/px2cosmo/mock/

# train poisson likelihood
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_poisson.py 'q_omegat_log1pN_z14_v1'\
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N20000_poisson.v1.npy \
#    --zbin 'z14'\
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_poisson.py 'q_omegat_log1pN_z11_v1'\
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N20000_poisson.v1.npy \
#    --zbin 'z11'\
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_poisson.py 'q_omegat_log1pN_z9_v1'\
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N20000_poisson.v1.npy \
#    --zbin 'z9'\
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --njobs 5 --cpu --verbose


# train population likelihood
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_pop.py 'q_X_omegasig_z14_v1_pop_k10' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_pop_N10000_k10_z14.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 50 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_pop.py 'q_X_omegasig_z11_v1_pop_k10' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_pop_N10000_k10_z11.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 50 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_pop.py 'q_X_omegasig_z9_v1_pop_k100' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_pop_N10000_k100_z9.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 512 --njobs 5 --cpu --verbose

#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_likelihood.py 'q_X_omegasig_z14_v1' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z14.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 50 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_likelihood.py 'q_X_omegasig_z11_v1' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z11.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 50 --njobs 5 --cpu --verbose
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nde_likelihood.py 'q_X_omegasig_z9_v1' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_N2000_z9.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 50 --njobs 5 --cpu --verbose


#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python dm_pop.py 'q_omega_Xsig_z11_nse' \
#    --training-data-file /Users/ch54662/data/px2cosmo/mock/mock_pop_N20000_z11.v1.npy \
#    --study-dir /Users/ch54662/data/px2cosmo/mock/ndes/ \
#    --batch-size 512 --njobs 5 --cpu --verbose

'''

script to train NPE for the poisson term  


'''
import os, sys 
import numpy as np 

import torch
import optuna

from px2cosmo import util as UT

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# input 
study_name  = sys.argv[1]
z_bin       = sys.argv[2] 
outdir      = sys.argv[3]

# optuna settings
n_trials    = 50
n_startup_trials = 20 
n_jobs      = 1
output_dir  = os.path.join(outdir, 'nde') 
os.system('mkdir -p %s' % os.path.join(output_dir, study_name))  
storage     = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)

# cpu/gpu
seed = 12387
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
  device = "cuda"
elif torch.backends.mps.is_available():
  device = "mps"
else:
  device = "cpu"

# load data 
_data = np.loadtxt("/Users/ch54662/data/px2cosmo/mock/mock_N2000_%s.dat" % z_bin, skiprows=1)
omegas  = _data[:,:4]
Xs      = _data[:,4:6]
sigs    = _data[:,6:]
Nmock   = omegas.shape[0]

if Nmock < 5000000: 
    ishuffle = np.arange(Nmock)
    np.random.shuffle(ishuffle)
else: 
    ishuffle = np.random.choice(np.arange(Nmock), 5000000)

_omegas = torch.tensor(np.hstack([omegas, sigs])[ishuffle].astype(np.float32)).to(device)
_Xs     = torch.tensor(Xs[ishuffle].astype(np.float32)).to(device)


def Objective(trial): 
    # hyperparameters
    nde_model   = trial.suggest_categorical("nde_model", ['zuko_maf', 'zuko_nsf'])
    n_transf    = trial.suggest_int("n_transf", 3, 7)
    n_hidden    = trial.suggest_int("n_hidden", 64, 256, log=True)
    n_bins      = trial.suggest_int("n_bins", 5, 10)

    # density estimator
    nde = posterior_nn(nde_model,
                       hidden_features=n_hidden,
                       num_transforms=n_transf,
                       num_bins=n_bins)

    # neural inference  
    inference = NPE(density_estimator=nde, device=device)
    _ = inference.append_simulations(_Xs, _omegas).train()
    p_X_omegasig = inference.build_posterior()

    # save trained NPE  
    fmodel  = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(p_X_omegasig, fmodel)
        
    # best validation loss 
    best_valid_log_prob = inference._summary['best_validation_loss'][-1]
    return best_valid_log_prob 


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, direction="minimize", load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)

'''

script to train NDEs


'''
import os, sys 
import numpy as np 

import torch
import optuna

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# input 
study_name  = sys.argv[1]
Nmocks  = int(sys.argv[2]) 
fdown   = int(sys.argv[3])  
outdir  = sys.argv[4]

# optuna settings
n_trials    = 1000
n_startup_trials = 20 
n_jobs      = 1
output_dir  = os.path.join(outdir, 'nde') 
os.system('mkdir -p %s' % os.path.join(output_dir, study_name))  
storage     = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)

# cpu/gpu
seed = 12387
torch.manual_seed(seed)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(seed)
device = ("cuda" if cuda else "cpu")

# load data 
params  = np.load(os.path.join(outdir, "mock0_N%ifdown%i.params.npy" % (Nmocks, fdown)))
data    = np.load(os.path.join(outdir, "mock0_N%ifdown%i.data.npy" % (Nmocks, fdown)))


# shuffle data 
ishuffle = np.arange(params.shape[0])
np.random.shuffle(ishuffle)

_phi = torch.tensor(params[ishuffle][:int(params.shape[0]*0.9)].astype(np.float32)).to(device)
_X = torch.tensor(data[ishuffle][:int(params.shape[0]*0.9)].astype(np.float32)).to(device)


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
    _ = inference.append_simulations(_X, _phi).train()

    p_X_phi = inference.build_posterior()

    # save trained NPE  
    fmodel  = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(p_X_phi, fmodel)
        
    # best validation loss 
    best_valid_log_prob = inference._summary['best_validation_loss'][0]
    return -1*best_valid_log_prob 


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, direction="minimize", load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)

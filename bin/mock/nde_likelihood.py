'''

script to train NPE for the poisson term  


'''
import os
import argparse
import numpy as np 

import torch
import optuna

from px2cosmo import util as UT

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn


def parse_args(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_name", help='optuna study name') 
    parser.add_argument("--training-data-file", required=True, 
            help="training data")
    parser.add_argument("--study-dir", required=True, 
            help="directory to save the optuna study")
    parser.add_argument("--njobs", type=int, default=1, 
            help="number of optuna jobs") 
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


if __name__=="__main__": 
    # parse arguments 
    args = parse_args()

    # optuna settings
    n_trials            = 1000 
    n_startup_trials    = 20 
    n_jobs              = args.njobs
    os.system('mkdir -p %s' % os.path.join(args.study_dir, args.study_name))  
    storage     = 'sqlite:///%s/%s/%s.db' % (args.study_dir, args.study_name, args.study_name)
    if args.verbose: print('writing ndes to %s' % os.path.join(args.study_dir, args.study_name))

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
    if args.verbose: print(f'device: {device}') 

    # load data 
    _data = np.load(args.training_data_file)
    omegas  = _data[:,:4]
    Xs      = _data[:,4:6]
    sigs    = _data[:,6:]
    Nmock   = omegas.shape[0]
    if args.verbose: print(f'loading data from {args.training_data_file}') 

    if args.test: 
        max_Nmock = 500
    else: 
        max_Nmock = 5000000

    if Nmock < max_Nmock: 
        ishuffle = np.arange(Nmock)
        np.random.shuffle(ishuffle)
    else: 
        ishuffle = np.random.choice(np.arange(Nmock), max_Nmock, replace=False)

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
        _ = inference.append_simulations(_Xs, _omegas).train(
                training_batch_size=512, 
                learning_rate=5e-3)
        p_X_omegasig = inference.build_posterior()

        # save trained NPE  
        fmodel  = os.path.join(args.study_dir, args.study_name, '%s.%i.pt' % (args.study_name, trial.number))
        torch.save(p_X_omegasig, fmodel)
            
        # best validation loss 
        best_valid_log_prob = inference._summary['best_validation_loss'][-1]
        return best_valid_log_prob 


    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study   = optuna.create_study(study_name=args.study_name, 
            sampler=sampler, 
            storage=storage, 
            direction="minimize", 
            load_if_exists=True)

    study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)

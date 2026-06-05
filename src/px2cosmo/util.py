'''
useful functions
'''
import os
import torch
import optuna


def best_model(study_name, output_dir):
    storage = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
    study   = optuna.load_study(study_name=study_name, storage=storage)
    best    = study.best_trial
    fmodel  = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, best.number))
    return torch.load(fmodel), best


def LF(Muv, z, phi):
    alpha, beta, gamma, Muv_s = phi
    return (10**(gamma * (z - 9.)))/(10**(0.4*(alpha + 1)*(Muv - Muv_s) + 10**(0.4*(beta + 1)*(Muv - Muv_s))))

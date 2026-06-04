'''

make training data 

'''
import os, sys
import numpy as np 
from px2cosmo import fm as FM 

Nmocks = int(sys.argv[1]) 

# sample LF parameters 
# phi_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
alphas  = np.random.uniform(-1.8, -1.5, size=Nmocks)
betas   = np.random.uniform(-1.8, -1.2, size=Nmocks)
gammas  = np.random.uniform(-0.4, 0, size=Nmocks)
Muvss   = np.random.uniform(-22., -18., size=Nmocks)


phis, all_mocks_nonoise, all_mocks = [], [], []  
for i in range(Nmocks): 
    phi = np.array([alphas[i], betas[i], gammas[i], Muvss[i]])

    # sample LF 
    mock = FM.sampleLF(phi, phi_amp=6e-3)
    
    # survey selection  
    select = FM.selection_function(mock, name='mock0') 
    all_mocks.append(mock[select]) 
    phis.append(np.tile(phi, (np.sum(select), 1)))

np.save('/Users/chang/data/px2cosmo/test0/mock0_data.npy', 
        np.concatenate(all_mocks, axis=0))
np.save('/Users/chang/data/px2cosmo/test0/mock0_params.npy', 
        np.concatenate(phis, axis=0))

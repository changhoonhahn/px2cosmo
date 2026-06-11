'''

make training data 

'''
import os, sys
import numpy as np 
from px2cosmo import fm as FM 

Nmocks  = int(sys.argv[1]) 
outdir  = sys.argv[2]

# sample LF parameters 
# phi_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
alphas  = np.random.uniform(-1.8, -1.5, size=Nmocks)
betas   = np.random.uniform(-1.8, -1.2, size=Nmocks)
gammas  = np.random.uniform(-0.4, 0, size=Nmocks)
Muvss   = np.random.uniform(-22., -18., size=Nmocks)


phis, counts = [], []
for i in range(Nmocks): 
    phi = np.array([alphas[i], betas[i], gammas[i], Muvss[i]])
    phis.append(phi)

    # froward model 
    mock = FM.forwardmodel(phi, name='test0')
    
    counts.append(mock.shape[0]) 

np.save(os.path.join(outdir, 'mock0_N%i.counts.npy' % Nmocks), np.array(counts))
np.save(os.path.join(outdir, 'mock0_N%i.counts_params.npy' % Nmocks), np.array(phis))

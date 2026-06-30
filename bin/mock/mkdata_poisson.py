'''

make training data with homoskedastic noise 

'''
import os, sys
import numpy as np 
from tqdm import tqdm 
from px2cosmo import fm as FM 

Nmocks  = int(sys.argv[1]) 
outdir  = sys.argv[2]

# sample LF parameters 
# omega_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
# phi_amp = 6e-3 
omegas = np.array([
    np.random.uniform(-1.8, -1.5, size=Nmocks), 
    np.random.uniform(-1.8, -1.2, size=Nmocks),
    np.random.uniform(-0.4, 0, size=Nmocks), 
    np.random.uniform(-22., -18., size=Nmocks), 
    np.random.uniform(1, 10, size=Nmocks) * 1e-3
]).T



fdata = os.path.join(outdir, 'mock_N%i_poisson.dat' % Nmocks)
with open(fdata, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, phi_amp, N(z=14), N(z=11), N(z=9)\n")

for i in tqdm(range(Nmocks)): 
    # forward model 
    mock0 = FM.forwardmodel(omegas[i,:-1], name='z14', phi_amp=omegas[i,-1])
    mock1 = FM.forwardmodel(omegas[i,:-1], name='z11', phi_amp=omegas[i,-1])
    mock2 = FM.forwardmodel(omegas[i,:-1], name='z9', phi_amp=omegas[i,-1])

    with open(fdata, "a") as f:
        f.write(" ".join(map(str, omegas[i])) + " " +  
                " ".join(map(str, [mock0.shape[0], mock1.shape[0], mock2.shape[0]])) + "\n")

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
    np.random.uniform(-22., -18., size=Nmocks) 
]).T


fdata0 = os.path.join(outdir, 'mock_N%i_z14.dat' % Nmocks)
with open(fdata0, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, z, Muv, sig_z, sig_Muv \n")

fdata1 = os.path.join(outdir, 'mock_N%i_z11.dat' % Nmocks)
with open(fdata1, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, z, Muv, sig_z, sig_Muv \n")

fdata2 = os.path.join(outdir, 'mock_N%i_z9.dat' % Nmocks)
with open(fdata2, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, z, Muv, sig_z, sig_Muv \n")


for i in tqdm(range(Nmocks)): 
    # forward model 
    mock0 = FM.forwardmodel(omegas[i], name='z14', phi_amp=6e-3)
    mock1 = FM.forwardmodel(omegas[i], name='z11', phi_amp=6e-3)
    mock2 = FM.forwardmodel(omegas[i], name='z9', phi_amp=6e-3)

    with open(fdata0, "a") as f:
        for row in mock0:
            f.write(" ".join(map(str, omegas[i])) + " " + " ".join(map(str, row)) + "\n")

    with open(fdata1, "a") as f:
        for row in mock1:
            f.write(" ".join(map(str, omegas[i])) + " " + " ".join(map(str, row)) + "\n")

    with open(fdata2, "a") as f:
        for row in mock2:
            f.write(" ".join(map(str, omegas[i])) + " " + " ".join(map(str, row)) + "\n")

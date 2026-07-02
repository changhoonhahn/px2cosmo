'''

make training data with homoskedastic noise 

'''
import os, sys
import numpy as np 
from tqdm import tqdm 
from px2cosmo import fm as FM 
from px2cosmo import util as UT 

Nmocks  = int(sys.argv[1]) 
outdir  = sys.argv[2]

bounds = UT._prior_range_default()

# sample LF parameters 
# omega_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
# phi_amp = 6e-3 
omegas = np.array([
    np.random.uniform(bounds[0][0], bounds[0][1], size=Nmocks), 
    np.random.uniform(bounds[1][0], bounds[1][1], size=Nmocks),
    np.random.uniform(bounds[2][0], bounds[2][1], size=Nmocks), 
    np.random.uniform(bounds[3][0], bounds[3][1], size=Nmocks)
]).T


fdata0 = os.path.join(outdir, 'mock_N%i_z14.v1.dat' % Nmocks)
with open(fdata0, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, z, Muv, sig_z, sig_Muv \n")

fdata1 = os.path.join(outdir, 'mock_N%i_z11.v1.dat' % Nmocks)
with open(fdata1, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs, z, Muv, sig_z, sig_Muv \n")

fdata2 = os.path.join(outdir, 'mock_N%i_z9.v1.dat' % Nmocks)
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

# save to numpy for faster I/O
data = np.loadtxt(fdata0, skiprows=1) 
np.save(fdata0.replace('.dat', '.npy'), data) 

data = np.loadtxt(fdata1, skiprows=1) 
np.save(fdata1.replace('.dat', '.npy'), data) 

data = np.loadtxt(fdata2, skiprows=1) 
np.save(fdata2.replace('.dat', '.npy'), data) 


'''

make training data with homoskedastic noise 

'''
import os, sys
import numpy as np 
from px2cosmo import fm as FM 

Nmocks  = int(sys.argv[1]) 
Nsub    = int(sys.argv[2])  
outdir  = sys.argv[3]

# sample LF parameters 
# phi_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
# phi_amp = 6e-3 
alphas  = np.random.uniform(-1.8, -1.5, size=Nmocks)
betas   = np.random.uniform(-1.8, -1.2, size=Nmocks)
gammas  = np.random.uniform(-0.4, 0, size=Nmocks)
Muvss   = np.random.uniform(-22., -18., size=Nmocks)
phiamps = np.random.uniform(1, 10, size=Nmocks) * 1e-3

fdata           = os.path.join(outdir, 'mock3_N%iNsub%i.data.dat' % (Nmocks, Nsub))
fparam          = os.path.join(outdir, 'mock3_N%iNsub%i.params.dat' % (Nmocks, Nsub))

fcounts         = os.path.join(outdir, 'mock3_N%i.counts.dat' % Nmocks)
fcounts_param   = os.path.join(outdir, 'mock3_N%i.counts_params.dat' % Nmocks)

with open(fdata, "w") as f:
    f.write("# z, Muv, sig_z, sig_Muv, phi_amp\n")

with open(fparam, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs \n")

with open(fcounts, "w") as f:
    f.write("# counts \n")

with open(fcounts_param, "w") as f:
    f.write("# alpha, beta, gamma, Mnuvs \n")


phis, all_mocks_nonoise, all_mocks = [], [], []  
phis_counts, counts = [], []  
for i in range(Nmocks): 
    phi = np.array([alphas[i], betas[i], gammas[i], Muvss[i]])

    # forward model 
    mock = FM.forwardmodel(phi, name='test2', phi_amp=phiamps[i])

    with open(fcounts, "a") as f:
        f.write("%i\n" % mock.shape[0])

    with open(fcounts_param, "a") as f:
        row = np.concatenate([phi, [phiamps[i]]]) 
        f.write(" ".join(map(str, row)) + "\n")
    
    # subsample Nsub data points  
    mock = FM.forwardmodel(phi, name='test2')

    _N = mock.shape[0]
    irandom = np.random.choice(np.arange(_N), size=Nsub)

    with open(fdata, "a") as f:
        for row in mock[irandom]:
            f.write(" ".join(map(str, row)) + "\n")

    with open(fparam, "a") as f:
        for _ in range(Nsub): 
            row = phi 
            f.write(" ".join(map(str, row)) + "\n")
    

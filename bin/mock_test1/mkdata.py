'''

make training data with homoskedastic noise 

'''
import os, sys
import numpy as np 
from px2cosmo import fm as FM 

Nmocks  = int(sys.argv[1]) 
fdown   = int(sys.argv[2])  
outdir  = sys.argv[3]

# sample LF parameters 
# phi_true = np.array([-1.65, -1.5, -0.2, -19.5]) # fit by eye to zeus21 output
alphas  = np.random.uniform(-1.8, -1.5, size=Nmocks)
betas   = np.random.uniform(-1.8, -1.2, size=Nmocks)
gammas  = np.random.uniform(-0.4, 0, size=Nmocks)
Muvss   = np.random.uniform(-22., -18., size=Nmocks)


fdata = os.path.join(outdir, 'mock0_N%ifdown%i.data.dat' % (Nmocks, fdown))
fparam = os.path.join(outdir, 'mock0_N%ifdown%i.params.dat' % (Nmocks, fdown))

fcounts = os.path.join(outdir, 'mock1_N%i.counts.dat' % Nmocks)
fcounts_param = os.path.join(outdir, 'mock1_N%i.counts_params.dat' % Nmocks)

with open(fdata, "w") as f:
    f.write("# z, Mnu \n")

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
    mock = FM.forwardmodel(phi, name='test1')

    with open(fdata, "a") as f:
        for row in mock[::fdown]:
            f.write(" ".join(map(str, row)) + "\n")

    with open(fparam, "a") as f:
        for _ in np.arange(mock.shape[0])[::fdown]:
            row = phi 
            f.write(" ".join(map(str, row)) + "\n")
    
    with open(fcounts, "a") as f:
        f.write("%i\n" % mock.shape[0])

    with open(fcounts_param, "a") as f:
        row = phi 
        f.write(" ".join(map(str, row)) + "\n")

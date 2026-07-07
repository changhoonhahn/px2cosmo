'''

make training data for the Poisson Likelihood NDE 

'''
import os, sys
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from px2cosmo import fm as FM
from px2cosmo import util as UT


def _worker(omega):
    mock0 = FM.forwardmodel(omega[:-1], name='z14', phi_amp=omega[-1])
    mock1 = FM.forwardmodel(omega[:-1], name='z11', phi_amp=omega[-1])
    mock2 = FM.forwardmodel(omega[:-1], name='z9',  phi_amp=omega[-1])
    return omega.tolist() + [mock0.shape[0], mock1.shape[0], mock2.shape[0]]


if __name__ == '__main__':
    Nmocks  = int(sys.argv[1])
    outdir  = sys.argv[2]

    bounds = UT._prior_range_default()

    omegas = np.array([
        np.random.uniform(bounds[0][0], bounds[0][1], size=Nmocks),
        np.random.uniform(bounds[1][0], bounds[1][1], size=Nmocks),
        np.random.uniform(bounds[2][0], bounds[2][1], size=Nmocks),
        np.random.uniform(bounds[3][0], bounds[3][1], size=Nmocks),
        np.random.uniform(bounds[4][0], bounds[4][1], size=Nmocks)
    ]).T

    with Pool() as pool:
        rows = list(tqdm(pool.imap(_worker, omegas), total=Nmocks))

    data = np.array(rows)

    fdata = os.path.join(outdir, 'mock_N%i_poisson.v1.dat' % Nmocks)
    with open(fdata, "w") as f:
        f.write("# alpha, beta, gamma, Mnuvs, phi_amp, N(z=14), N(z=11), N(z=9)\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")

    np.save(fdata.replace('.dat', '.npy'), data)


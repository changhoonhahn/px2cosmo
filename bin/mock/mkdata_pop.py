'''

make training data with for population NLE

'''
import os, sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from px2cosmo import fm as FM
from px2cosmo import util as UT


def _worker(omega):
    # Sample LF once; apply 3 redshift-bin selections from the same catalog
    mock = FM.sampleLF(omega, phi_amp=6e-3)
    results = []
    for name in ('z14', 'z11', 'z9'):
        obs = FM.apply_selection_noise(mock, name)  # (N, 4): z, Muv, sig_z, sig_Muv
        if obs.shape[0] == 0:
            results.append(None)
            continue
        i = np.random.randint(obs.shape[0])
        results.append(np.concatenate([omega, obs[i]]))  # (8,): alpha,beta,gamma,Muv_s,z,Muv,sig_z,sig_Muv
    return results


if __name__ == '__main__':
    Nmocks  = int(sys.argv[1])
    outdir  = sys.argv[2]

    bounds = UT._prior_range_default()

    omegas = np.array([
        np.random.uniform(bounds[0][0], bounds[0][1], size=Nmocks),
        np.random.uniform(bounds[1][0], bounds[1][1], size=Nmocks),
        np.random.uniform(bounds[2][0], bounds[2][1], size=Nmocks),
        np.random.uniform(bounds[3][0], bounds[3][1], size=Nmocks)
    ]).T

    with Pool() as pool:
        results = list(tqdm(pool.imap(_worker, omegas), total=Nmocks))

    for j, zname in enumerate(('z14', 'z11', 'z9')):
        rows = [r[j] for r in results if r[j] is not None]
        data = np.vstack(rows)  # (Nmocks, 8)
        fout = os.path.join(outdir, 'mock_pop_N%i_%s.v1.npy' % (Nmocks, zname))
        np.save(fout, data)

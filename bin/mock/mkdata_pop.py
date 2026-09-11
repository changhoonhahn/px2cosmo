'''

make training data with for population NLE

'''
import os, sys
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from px2cosmo import fm as FM
from px2cosmo import util as UT


def _worker(omega, k=1):
    results = []
    for name in ('z14', 'z11', 'z9'):
        # Each bin samples independently; accumulated to avoid inflated mocks
        collected = []
        n_collected = 0
        fup = 1.
        while n_collected < k:
            mock = FM.sampleLF(omega, phi_amp=6e-3 * fup)
            obs = FM.apply_selection_noise(mock, name)
            if obs.shape[0] > 0:
                collected.append(obs)
                n_collected += obs.shape[0]
            fup = min(fup * 2, 64.)  # cap amplification at 64x

        obs = np.vstack(collected)
        idx = np.random.choice(obs.shape[0], size=k, replace=False)
        rows = np.column_stack([np.tile(omega, (k, 1)), obs[idx]])  # (k, 8)
        results.append(rows)
    return results


if __name__ == '__main__':
    Nmocks  = int(sys.argv[1])
    outdir  = sys.argv[2]
    k       = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    nproc   = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    bounds = UT._prior_range_default()

    omegas = np.array([
        np.random.uniform(bounds[0][0], bounds[0][1], size=Nmocks),
        np.random.uniform(bounds[1][0], bounds[1][1], size=Nmocks),
        np.random.uniform(bounds[2][0], bounds[2][1], size=Nmocks),
        np.random.uniform(bounds[3][0], bounds[3][1], size=Nmocks)
    ]).T

    # Open output files upfront so we can stream results without holding all in RAM
    znames = ('z14', 'z11', 'z9')
    fouts = [
        os.path.join(outdir, 'mock_pop_N%i_k%i_%s.v1.npy' % (Nmocks, k, zname))
        for zname in znames
    ]
    buffers = [[] for _ in znames]

    worker = partial(_worker, k=k)
    with Pool(processes=nproc) as pool:
        for result in tqdm(pool.imap(worker, omegas, chunksize=10), total=Nmocks):
            for j in range(len(znames)):
                buffers[j].append(result[j])

    for j, fout in enumerate(fouts):
        data = np.vstack(buffers[j])
        np.save(fout, data)

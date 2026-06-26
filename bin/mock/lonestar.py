'''

generate and run slurm scripts for TACC Lonestar

'''
import os, sys
import numpy as np


def nde(zred, njobs=1, time=1, queue='development', silent=True):
    _dir= '/corral/utexas/AST25023/px2cosmo/test0/'
    scriptdir = os.path.dirname(__file__)
    
    hr = int(np.floor(time))
    mn = int((time * 60) % 60)
    ftrain = '/home1/11004/chahah/work/px2cosmo/mock/mock_N2000_z%i.npy' % zred
    study_dir = '/home1/11004/chahah/work/px2cosmo/mock/nde'

    fout = 'o/nle.mock_z%i.o' % zred 
    while os.path.isfile(os.path.join(scriptdir, fout)): 
        fout = fout.replace('.o', '_.o') 

    # write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J nle.mock_z%i' % (zred),
        '#SBATCH -o %s' % fout, 
        '#SBATCH -p %s' % queue, 
        '#SBATCH -N 1',               
        '#SBATCH -n 1',               
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST26017', 
        '',
        "module purge ",
        "module load intel",  
        "", 
        "unset PYTHONPATH", 
        "source ~/.bashrc", 
        "", 
        "conda activate jwst",
        '',
        'python -u %s/nde_likelihood.py q_X_omegasig_z%i --training-data-file %s --study-dir %s --batch-size 512 --njobs %i --verbose' % (scriptdir, zred, ftrain,
        study_dir, njobs), 
        ''])
    
    # create the script.sh file, execute it and remove it
    f = open(os.path.join(os.environ['WORK'], 'script.slurm'),'w')
    f.write(a)
    f.close()
    os.system('sbatch %s' % os.path.join(os.environ['WORK'], 'script.slurm'))
    #os.system('rm script.slurm')
    return None


if __name__=="__main__":
    nde(7, njobs=10, time=10, queue='normal', silent=True)
    nde(9, njobs=10, time=10, queue='normal', silent=True)
    nde(11, njobs=10, time=10, queue='normal', silent=True)

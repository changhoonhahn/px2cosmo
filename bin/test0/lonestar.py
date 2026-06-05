'''

generate and run slurm scripts for TACC Lonestar

'''
import os, sys
import numpy as np


def mkdata(Nphi, fdown, time=1, queue='development', silent=True):
    ''' run ALPT for specific LHC realization 
    '''
    _dir= '/corral/utexas/AST25023/px2cosmo/test0/'
    scriptdir = os.path.dirname(__file__)
    
    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    # write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J mkdata.test0',
        '#SBATCH -o o/mkdata.test0',
        '#SBATCH -p %s' % queue, 
        '#SBATCH -N 1',               
        '#SBATCH -n 1',               
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST25022', 
        '',
        "module purge ",
        "module load intel",  
        "", 
        "unset PYTHONPATH", 
        "source ~/.bashrc", 
        "", 
        "conda activate jwst",
        '',
        'python %s/mkdata.py %i %i %s' % (scriptdir, Nphi, fdown, _dir),  
        ''])
    
    # create the script.sh file, execute it and remove it
    f = open(os.path.join(os.environ['WORK'], 'script.slurm'),'w')
    f.write(a)
    f.close()
    os.system('sbatch %s' % os.path.join(os.environ['WORK'], 'script.slurm'))
    #os.system('rm script.slurm')
    return None


def nde(study, Nphi, fdown, time=1, queue='development', silent=True):
    _dir= '/corral/utexas/AST25023/px2cosmo/test0/'
    scriptdir = os.path.dirname(__file__)
    
    hr = int(np.floor(time))
    mn = int((time * 60) % 60)

    # write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J nde.test0.%s' % study,
        '#SBATCH -o o/nde.test0.%s' % study,
        '#SBATCH -p %s' % queue, 
        '#SBATCH -N 1',               
        '#SBATCH -n 1',               
        '#SBATCH --time=%s:%s:00' % (str(hr).zfill(2), str(mn).zfill(2)),
        '#SBATCH -A AST25022', 
        '',
        "module purge ",
        "module load intel",  
        "", 
        "unset PYTHONPATH", 
        "source ~/.bashrc", 
        "", 
        "conda activate jwst",
        '',
        'python %s/nde.py %s %i %i %s' % (scriptdir, study, Nphi, fdown, _dir),  
        ''])
    
    # create the script.sh file, execute it and remove it
    f = open(os.path.join(os.environ['WORK'], 'script.slurm'),'w')
    f.write(a)
    f.close()
    os.system('sbatch %s' % os.path.join(os.environ['WORK'], 'script.slurm'))
    #os.system('rm script.slurm')
    return None


if __name__=="__main__":
    #mkdata(100000, 10, time=5, queue='normal', silent=True)# generate 100,000 mocks but downsample the galaxies by 10x
    nde('test0', 100, 10, time=0.5, queue='development', silent=True)

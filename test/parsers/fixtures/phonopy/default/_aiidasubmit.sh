#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N aiida-187250
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe mpi* 24
#$ -l h_rt=10:00:00

'/home/togo/.miniconda/envs/dev/bin/phonopy' '-c' 'phonopy_params.yaml' '--pdos=auto' '--mesh=50.000000' '--nowritemesh' '--writefc' '--writefc-format=hdf5' '--nac'

'/home/togo/.miniconda/envs/dev/bin/phonopy' '-c' 'phonopy_params.yaml' '-t' '--mesh=50.000000' '--nowritemesh' '--readfc' '--readfc-format=hdf5' '--nac'

'/home/togo/.miniconda/envs/dev/bin/phonopy' '-c' 'phonopy_params.yaml' '--band=auto' '--band-points=101' '--band-const-interval' '--readfc' '--readfc-format=hdf5' '--nac'

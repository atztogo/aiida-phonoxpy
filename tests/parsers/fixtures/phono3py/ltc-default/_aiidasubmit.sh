#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m n
#$ -N aiida-226060
#$ -V
#$ -o _scheduler-stdout.txt
#$ -e _scheduler-stderr.txt
#$ -pe mpi* 24

'/home/togo/.miniconda/envs/dev/bin/phono3py' '-c' 'phono3py_params.yaml.xz' '--fc2' '--fc3' '--mesh' '30' '--br' '--ts' '300' '--nac'

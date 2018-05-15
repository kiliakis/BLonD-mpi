#!/bin/bash
#SBATCH --time=2
#SBATCH --partition=be-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=setup-job.txt
#SBATCH --error=setup-job.txt
#SBATCH --job-name=setup-job

source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
cd $BLOND
# echo $BLOND

python --version
mpicc --version
# echo $PYTHONPATH

python setup_cpp.py -p
# export PYTHONPATH=$BLOND:$PYTHONPATH
# export OMP_NUM_THREADS=2
# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
# mpiexec -n 1 python -m mpi4py $2 $3
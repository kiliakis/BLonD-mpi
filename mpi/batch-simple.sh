#!/bin/bash
## SBATCH --time=45
## SBATCH --partition=be-long
## SBATCH --nodes=2
## SBATCH --ntasks=4
## SBATCH --ntasks-per-node=2
## SBATCH --cpus-per-task=1
## SBATCH --ntasks-per-core=1
#SBATCH --hint=nomultithread
## SBATCH --output="%x".txt
## SBATCH --error="%x".txt
## SBATCH --job-name="%x"

source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
# echo $BLOND

python --version
mpicc --version
# echo $PYTHONPATH

# python setup_cpp.py -p
export PYTHONPATH=$BLOND:$PYTHONPATH
# export OMP_NUM_THREADS=2
# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
mpiexec -n 1 python -m mpi4py $2 $3
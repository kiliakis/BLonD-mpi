#!/bin/sh
#SBATCH --time=30
#SBATCH --partition=be-long
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --output=ex-01.out
#SBATCH --error=ex-01.out
#SBATCH --job-name=ex-01

source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
echo $BLOND

python --version
mpicc --version
echo $PYTHONPATH

# python setup_cpp.py
export PYTHONPATH=$BLOND:$PYTHONPATH

# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
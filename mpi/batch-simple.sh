#!/bin/bash
#SBATCH --time=30
#SBATCH --partition=be-long
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --output=ex-01.out
#SBATCH --error=ex-01.out
#SBATCH --job-name=ex-01

source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
echo $BLOND

python --version
mpicc --version
echo $PYTHONPATH

python setup_cpp.py -p
export PYTHONPATH=$BLOND:$PYTHONPATH
export OMP_NUM_THREADS=4
# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
#!/bin/sh
#SBATCH --time=5
#SBATCH --partition=be-short
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --output=ex-01.out
#SBATCH --error=ex-01.out
#SBATCH --job-name=ex-01
# hostname
# pwd
# ls -l

source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
echo $BLOND

python --version
mpicc --version
echo $PYTHONPATH

python setup_cpp.py
export PYTHONPATH=$BLOND:$PYTHONPATH

python $BLOND/mpi/EX_01_Acceleration-master.py
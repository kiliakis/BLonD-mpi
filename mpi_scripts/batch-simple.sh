#!/bin/bash
##Comment SBATCH --time=45
##Comment SBATCH --partition=be-long
##Comment SBATCH --nodes=2
##Comment SBATCH --ntasks=4
##Comment SBATCH --ntasks-per-node=2
##Comment SBATCH --cpus-per-task=1
##Comment SBATCH --ntasks-per-core=1
##Comment SBATCH --output="%x".txt
##Comment SBATCH --error="%x".txt
##Comment SBATCH --job-name="%x"
#SBATCH --hint=nomultithread
#SBATCH --export=ALL

# source $HOME/.bashrc

BLOND=$HOME/git/BLonD-mpi
# echo $BLOND

which python
python --version

which mpirun
mpirun --version
# mpicc --version

# python setup_cpp.py -p
export PYTHONPATH="$BLOND:$HOME/install/"
# export OMP_NUM_THREADS=$OMP_NUM_THREADS
# echo $PYTHONPATH
# echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
# export OMP_NUM_THREADS=2
# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
# mpiexec -n 1 python -m mpi4py $@
# mpirun $@
srun $@
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
#SBATCH --mem=0
##Comment SBATCH --overcommit

source $HOME/.bashrc

old_module=$(module list 2>&1 | grep -ohE "mpi/\w+/[0-9.]+")
new_module=$1; shift
if [ "$old_module" != "$new_module" ]; then
    echo -e "Unloading $old_module"
    module unload $old_module
    echo -e "Loading $new_module"
    module load $new_module
fi

python_path=$1; shift
if [[ $PATH =~ $python_path ]]; then
    :
else
    echo -e "Adding $python_path in PATH"
    export PATH=$python_path:$PATH
fi

# BLOND=$HOME/git/BLonD-mpi
# echo $BLOND
# source /cvmfs/projects.cern.ch/intelsw/psxe/linux/setup.sh
# source /cvmfs/projects.cern.ch/intelsw/psxe/linux/x86_64/2019/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
# export PATH=$HOME/install/anaconda3/bin:$PATH

echo -e "PYTHONPATH=$PYTHONPATH \n"
echo -e "PATH=$PATH \n"

echo -e "which python"
which python
# python --version
echo -e "which mpirun"
which mpirun
# mpirun --version

module list
# mpicc --version

# python setup_cpp.py -p
# export PYTHONPATH="$BLOND:$HOME/install/:$PYTHONPATH"
# export OMP_NUM_THREADS=$OMP_NUM_THREADS
# echo $PYTHONPATH
# echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
# export OMP_NUM_THREADS=2
# mpiexec -n 1 python -m mpi4py $BLOND/mpi/EX_01_Acceleration-master.py
# mpiexec -n 1 python -m mpi4py $@
# mpirun $@
# srun --cpu-bind=ldoms  $@
# srun --cpu-bind=ldoms mpirun $@
srun $@
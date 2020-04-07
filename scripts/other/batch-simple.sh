#!/bin/bash
#SBATCH --hint=nomultithread
#SBATCH --export=ALL
#SBATCH --mem=0


old_module=$(module list 2>&1 | grep -ohE "mpi/\w+/[0-9.]+")
new_module=$1; shift
if [ "$old_module" != "$new_module" ]; then
    echo -e "Unloading $old_module"
    module unload $old_module
    echo -e "Loading $new_module"
    module load $new_module
fi

# python_path=$1; shift
# if [[ $PATH =~ $python_path ]]; then
#     :
# else
#     echo -e "Adding $python_path in PATH"
#     export PATH=$python_path:$PATH
# fi

echo -e "PYTHONPATH=$PYTHONPATH \n"
echo -e "PATH=$PATH \n"

echo -e "which python"
which python
echo -e "which mpirun"
which mpirun

module list

srun $@
#!/bin/bash
#SBATCH --time=2
#SBATCH --partition=be-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=setup-job.txt
#SBATCH --error=setup-job.txt
#SBATCH --job-name=setup-job
#SBATCH --export=ALL


cd $BUILD_DIR

python blond/compile.py --with-fftw --with-fftw-threads --with-fftw-lib=$INSTALL_DIR/lib/ --with-fftw-header=$INSTALL_DIR/include/ -p

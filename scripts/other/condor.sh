#!/bin/bash

echo "HOME = $HOME"
echo "BLONDHOME = $BLONDHOME"

source $HOME/.bashrc
cd $BLONDHOME

echo "PATH = $PATH"
echo "PYTHONPATH = $PYTHONPATH"

which mpirun
which python

$@

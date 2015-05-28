#!/bin/sh
#PBS -N doc2vec
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=32:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 -u run/build_doc2vec.py

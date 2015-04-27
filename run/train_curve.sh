#!/bin/sh
#PBS -N gpu-info
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com

cd $PBS_O_WORKDIR

module load python3
modile load gcc
module load qt
module load cuda

# setup local python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 train_curve.py

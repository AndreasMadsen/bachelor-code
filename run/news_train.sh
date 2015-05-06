#!/bin/sh
#PBS -N sutskever-train
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 run/news_train.py

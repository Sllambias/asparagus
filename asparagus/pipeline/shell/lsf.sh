#!/bin/bash
#BSUB -q p1
#BSUB -J Test
#BSUB -R "span[hosts=1]"
#BSUB -n 10
#BSUB -gpu "num=1" 
#BSUB -W 5:59
#BSUB -R "rusage[mem=30GB]"
#BSUB -o out_%J.out
#BSUB -e out_%J.out

source /dtu/p1/seblla/miniconda3/etc/profile.d/conda.sh
conda activate asparagus
nvidia-smi
lscpu
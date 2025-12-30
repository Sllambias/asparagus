#!/bin/bash
bsub <<EOT
#!/bin/bash
#BSUB -q p1
#BSUB -J EvalBox
#BSUB -R "span[hosts=1]"
#BSUB -n $1
##BSUB -gpu "num=$2" 
#BSUB -W 12:59
#BSUB -R "rusage[mem=5GB]"
#BSUB -o out_%J.out
#BSUB -e out_%J.out

nvidia-smi
$3
$4
EOT
#!/bin/bash 
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=EvalBox
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=$1
#SBATCH -p gpu --gres=gpu:$2
#SBATCH --time=12:59:00
#SBATCH --mem=100GB
#SBATCH --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl

nvidia-smi
$3
$4
EOT
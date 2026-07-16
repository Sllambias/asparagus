#!/bin/bash
# Submits one EvalBox job to Slurm. Called by asp_eval_box_run / asp_eval_box_prepare_data
# via get_run_cmd_for_scheduler — not meant to be invoked by hand.
#
# Positional arguments (filled in by the dispatcher):
#   $1  CPUs per task   (from the task's hardware config: num_workers)
#   $2  number of GPUs  (from the task's hardware config: num_devices)
#   $3  environment setup command ($ASPARAGUS_EVAL_BOX_ENV_CMD, e.g. "conda activate myenv")
#   $4  the asparagus command to run (e.g. "asp_finetune_seg --config-name=... +model=... +hardware=...")
#
# Cluster-specific settings are read from optional environment variables (put them
# in your .env next to the ASPARAGUS_* paths). Unset variables fall back to the
# defaults shown; ACCOUNT and EXCLUDE are omitted entirely when unset:
#   ASPARAGUS_SLURM_JOB_NAME    job name shown in squeue            (default: EvalBox)
#   ASPARAGUS_SLURM_PARTITION   partition to submit to              (default: gpu)
#   ASPARAGUS_SLURM_TIME        walltime limit                      (default: 12:59:00)
#   ASPARAGUS_SLURM_MEM         memory per job                      (default: 100GB)
#   ASPARAGUS_SLURM_ACCOUNT     account/project, if your cluster uses accounting
#   ASPARAGUS_SLURM_EXCLUDE     comma-separated nodes to avoid
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${ASPARAGUS_SLURM_JOB_NAME:-EvalBox}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$1
#SBATCH -p ${ASPARAGUS_SLURM_PARTITION:-gpu} --gres=gpu:$2
#SBATCH --time=${ASPARAGUS_SLURM_TIME:-12:59:00}
#SBATCH --mem=${ASPARAGUS_SLURM_MEM:-100GB}
${ASPARAGUS_SLURM_ACCOUNT:+#SBATCH --account=$ASPARAGUS_SLURM_ACCOUNT}
${ASPARAGUS_SLURM_EXCLUDE:+#SBATCH --exclude=$ASPARAGUS_SLURM_EXCLUDE}

nvidia-smi
$3
$4
EOT
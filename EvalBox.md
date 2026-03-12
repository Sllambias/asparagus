## How to run the Eval Box

Choose the Evaluation-Box config, the checkpoint ID to evaluate, and add model and hardware configs (if these are not embedded in the Evaluation-Box config)
```
asp_eval_box_prepare_data --config-name dev_box 
asp_eval_box_run --config-name dev_box checkpoint_run_id=49180 +model=unet_b_lw_dec
asp_eval_box_collect_results --config-name dev_box
```

The above command runs finetuning of the pretrained unet_b_lw_dec model with Run ID 49180. The EvalBox finetunes models for each finetuning CONFIG found the in the dev_box.yaml. This means each finetuning config must point to a dataset and include all required parameters for finetuning.

### Environment Variable 

Submitting jobs to HPCs often involve some environment setup. The ```ASPARAGUS_EVAL_BOX_ENV_CMD``` is a flexible Environment Variable that can contain multiple lines of environment setup. It will be run in the beginning of any job started using the eval box. For example, loading a conda environment could look like this ```ASPARAGUS_EVAL_BOX_ENV_CMD="source ~/miniconda3/etc/profile.d/conda.sh ; conda activate asparagus "```. And, if required, additional module loads or similar could be appended, as long as each line is separated by " ; ".

### Slurm

For Slurm (currently only supported scheduler) the Eval Box will automatically populate a job script ```submit_slurm.sh``` for each downstream task and submit them individually.

By default the ```submit_slurm.sh``` looks something like this but can and should be adapted if e.g. certain nodes should be selected or avoided:
```bash
#!/bin/bash 
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=EvalBox
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=$1
#SBATCH -p gpu --gres=gpu:$2
#SBATCH --time=12:59:00
#SBATCH --mem=30GB
#SBATCH --exclude=hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl

nvidia-smi
$3
$4
EOT
```

Using the example in the top $1 and $2 are read from the hardware config (12 CPUs and 1 GPU, respectively). $3 is read from the ```ASPARAGUS_EVAL_BOX_ENV_CMD``` environment variable to set up your environment (see [Enviroment Variable](#environment-variable)).  $4 is the Asparagus training command. The eval box will automatically select the appropriate one and populate it with the contents of the finetuning config.

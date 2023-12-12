#!/bin/bash -x

#SBATCH --account hai_oneprot
#SBATCH --output /p/project/hai_oneprot/merdivan1/oneprot/slurms/train_%j_out.out
#SBATCH --error /p/project/hai_oneprot/merdivan1/oneprot/slurms/train_%j_err.out
#SBATCH --time 08:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node 4
#SBATCH --nodes 2
#SBATCH --partition booster
#SBATCH --gres gpu:4


#echo 'SLURM JOB ID'
#echo $SLURM_JOB_ID
#echo 'Loading bashrc'
#source $HOME/.bashrc
#echo 'Starting training'


export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

CACHE_DIR=/p/scratch/hai_oneprot/huggingface
export TRANSFORMERS_CACHE="$CACHE_DIR"/models
export HF_DATASETS_CACHE="$CACHE_DIR"/datasets
export HF_MODULES_CACHE="$CACHE_DIR"/modules
export HF_METRICS_CACHE="$CACHE_DIR"/metrics
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}i"



#source /p/project/hai_oneprot/merdivan1/sc_venv_template/activate.sh
#ml shpc
#ml nvcr.io/nvidia/pytorch/23.07-py3/module
#HYDRA_FULL_ERROR=1 srun python -u src/train.py experiment=example_collate_ddp

module purge
module load Stages/2024  GCCcore/.12.3.0 Apptainer-Tools/2024
#module load Apptainer-Tools

#export PYTHONPATH=""
echo "-------- Inside of the container ------------"
#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif python3 src/train.py experiment=train_ddp'

srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c "source /p/project/hai_oneprot/merdivan1/sc_venv_template/venv/bin/activate && python src/train.py experiment=train_ddp"'
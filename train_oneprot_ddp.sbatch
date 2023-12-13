#!/bin/bash -x

#SBATCH --account hai_oneprot                                                      # specify the account name, otherwise the job will not run
#SBATCH --output /p/project/hai_oneprot/merdivan1/oneprot/slurms/train_%j_out.out  # specify the file for the slurm output; make sure to change it to your folder
#SBATCH --error /p/project/hai_oneprot/merdivan1/oneprot/slurms/train_%j_err.out   # specify the file for the slurm error message; make sure to change it to your folder
#SBATCH --time 00:15:00                                                            # pecify the running time; on develbooster it is limited to 2 hours
#SBATCH --cpus-per-task=24                                                         # do not change
#SBATCH --ntasks-per-node 4                                                        # number of tasks per node; each node has 4 GPUs
#SBATCH --nodes 1                                                                  # the number of nodes; if you request 4 nodes you get 16 GPUs
#SBATCH --partition develbooster                                                   # the walltime on develbooster is 2 hours, on booster 24 hours; maximum number of nodes on develbooster is 4 nodes
#SBATCH --gres gpu:4                                                               # do not change


######################## DO NOT CHANGE THE BELOW ################################

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


module purge
module load Stages/2024  GCCcore/.12.3.0 Apptainer-Tools/2024
#module load Apptainer-Tools


echo "-------- Inside of the container ------------"

##################################################################################

#The below script runs a job using the container only

#srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif python3 src/train.py experiment=train_ddp'

#The below script activates a virtual environment within the container

srun --cpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES="0,1,2,3"; export PYTHONPATH=""; export HYDRA_FULL_ERROR=1; apptainer run --nv /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif bash -c "source /p/project/hai_oneprot/merdivan1/sc_venv_template/venv/bin/activate && python src/train.py experiment=train_ddp"'

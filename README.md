

<div align="center">

# OneProt

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Getting started with Judoor

- If you do not have an account on JUDOOR, our portal, please register: https://judoor.fz-juelich.de/register
- Video instructions: https://drive.google.com/file/d/1-DfiNBP4Gta0av4lQmubkXIXzr2FW4a-/view
- Join our project  https://judoor.fz-juelich.de/projects/join/hai_oneprot
- Sign the usage agreements as shown on the video https://drive.google.com/file/d/1mEN1GmWyGFp75uMIi4d6Tpek2NC_X8eY/view
- To set up `ssh` you can follow instructions from these slides https://helmholtzai-fzj.github.io/2023-getting-started-with-ai-on-supercomputers/01-access-machines.html#/29
- Once you connect to juwels-booster the project folder is `/p/project/hai_oneprot` (or `$PROJECT_hai_oneprot`). You need to create a folder with your judoor username and can clone the repo there.
- The folder `/p/scratch/hai_oneprot/` contains the data and pretrained model. If you want to add any new pretrained model/data you should add it there.
- If you are creating a new folder within `/p/scratch/hai_oneprot/`, don't forget to give permissions with `chmod -R u+rwX,g+rwX /path/to/your/folder` 

## Submitting a job to the SLURM cluster

- `sbatch train_oneprot_ddp.sbatch` from the terminal
- Look at `train_oneprot_ddp.sbatch` for comments on the SLURM parameters
- After you run the job the logs will be under `/p/project/hai_oneprot/your_username/oneprot/logs/`. For training runs under `train` and for evaluation under `eval` with the submission date and time.
Inside the this folder you can see the configs, logger outputs and checkpoints.

## Getting started with WandB

- Sign up https://wandb.ai/site
- Join the project https://wandb.ai/oneprot?invited=&newUser=false (or send your username to Erinc)
- In order to push your runs to WandB, please first activate the environment inside the container:

`apptainer run /p/project/hai_oneprot/merdivan1/singularity_docker_jupyter/singularity_docker_jupyter.sif`
`source /p/project/hai_oneprot/merdivan1/sc_venv_template/activate`
- Then run `wandb sync --include-offline -e oneprot logs/train/runs/name_of_your_run/wandb/offline-*`

## Getting started with Lightning and Hydra
- https://lightning.ai/docs/pytorch/stable/tutorials.html lightning tutorial
- https://hydra.cc/docs/tutorials/intro/ hydra tutorial

## Description

This project is dedicated to advancing the understanding and application of various modalities related to proteins, starting with sequence, structure and multiple sequence alignments (MSA). We will add more modalities along the way such as GO, Text, Gene, PPI, MD and more. 

We are aiming to learn aligned embeddings for different protein modalities. These different modalities can later be used on retrieval, prediction and generation tasks for proteins. 

## Modalities 

- Sequence
- Structure
- MSA
- upcoming

<br>

## Dataset 
We only require paired modalities dataset. 
| Modality 1 | Modality 2 | Dataset Size (Train/Val/Test) |
|----------|----------|----------|
| Sequence | Structure | 794057 / 9002 / 8801 |
| Sequence | MSA | 794057 / 9002 / 8801 |



<br>

## Main Ideas


- [**ImageBind**](https://arxiv.org/abs/2305.05665)
- [**CLIP**](https://arxiv.org/abs/2103.00020)

DownStream Tasks:

- [**SaProt**](https://www.biorxiv.org/content/10.1101/2023.10.01.)
<br>

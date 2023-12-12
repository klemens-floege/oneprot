

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

## Getting started with WandB

- Sign up https://wandb.ai/site
- Join the project https://wandb.ai/oneprot?invited=&newUser=false (or send your username to Erinc)

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

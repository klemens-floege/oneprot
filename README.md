

<div align="center">

More details coming soon

# OneProt

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Getting started with Lightning and Hydra
- https://lightning.ai/docs/pytorch/stable/tutorials.html lightning tutorial
- https://hydra.cc/docs/tutorials/intro/ hydra tutorial

## Description

This project is dedicated to advancing the understanding and application of various modalities related to proteins, such as sequence, structure, represented as graphs and as foldseek tokens, pockets and sequence similarity tuples, based on mutational information and multiple sequence alignments (MSA). 

We are aiming to learn aligned embeddings for different protein modalities. These different modalities can later be used on retrieval, prediction and generation tasks for proteins. 

## Modalities 

- Sequence
- Structure
- Text
- Pockets
- Sequence similarity

<br>

## Dataset 
We only require paired modalities dataset. 
| Modality 1 | Modality 2 | Dataset Size (Train/Val/Test) |
|----------|----------|----------|
| Sequence | Structure Graph | 647781 / 1000 / 1000 |
| Sequence | Structure Token | 1000000 / 1000 / 1000 |
| Sequence | Text | 540077 / 1000 / 1000 |
| Sequence | Pockets | 335086 / 1000 / 1000|
| Sequence | Sequence similarity| 1040560 / 1000 / 1000|



<br>

## Main Ideas


- [**ImageBind**](https://arxiv.org/abs/2305.05665)
- [**CLIP**](https://arxiv.org/abs/2103.00020)

DownStream Tasks:

- [**SaProt**](https://www.biorxiv.org/content/10.1101/2023.10.01.)
<br>

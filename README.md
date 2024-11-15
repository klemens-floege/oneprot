

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
### Dataset curation

We used OpenProteinSet [1], which contains structures, sequences, and MSAs for proteins from the PDB [2] and proteins from UniClust30 [3] and UniProtKB/Swiss-Prot [4]. We used MMseqs2 [5], to cluster the sequences with a sequence identity cut-off of 50\%, such that each cluster represents a homologous cluster in the protein fold space. We aligned the training, validation, and test splits along these sequence clusters. For each cluster representative and member, using the sequence, we find the structure from the AlphaFold2DB [6], the MSA from the OpenProteinSet, and the binding pocket with P2Rank [7]. As we could not find an MSA and binding pocket for each protein, fewer data points for these modalities are available. Sequence similarity dataset was constructed using ClinVar variant summary data [8] and MSA data. Each data-point in the sequence similarity dataset consists of three pairs of sequences corresponding to the same protein: original sequence and sequence with a benign mutation, two distinct pathogenic sequences, original sequence and a sequence sampled from the corresponding MSA. Such dataset enforces clustering of the proteins based on their biological relevance, e.g. moves pathogenic mutations away from benign ones.

| Modality 1 | Modality 2 | Dataset Size (Train/Val/Test) |
|----------|----------|----------|
| Sequence | Structure Graph | 647781 / 1000 / 1000 |
| Sequence | Structure Token | 1000000 / 1000 / 1000 |
| Sequence | Text | 540077 / 1000 / 1000 |
| Sequence | Pockets | 335086 / 1000 / 1000|
| Sequence | Sequence similarity| 1040560 / 1000 / 1000|

[1] Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Daniel Berenberg, Ian Fisk, Andrew M. Watkins, Stephen Ra, Richard Bonneau, and Mohammed AlQuraishi. Openproteinset: Training data for structural biology at scale, 2023b. URL https://arxiv.org/abs/2308.05326.\\
[2] Stephen K Burley, Charmi Bhikadiya, Chunxiao Bi, Sebastian Bittrich, Henry Chao, Li Chen, Paul A Craig, Gregg V Crichlow, Kenneth Dalenberg, Jose M Duarte, et al. Rcsb protein data bank (rcsb. org): delivery of experimentally-determined pdb structures alongside one million computed structure models of proteins from artificial intelligence/-machine learning. Nucleic acids research, 51(D1):D488–D508, 2023.
[3] Mirdita M., von den Driesch L., Galiez C., Martin M. J., Söding J., and Steinegger M. Uniclust databases of clustered and deeply annotated protein sequences and alignment. Nucleic Acids Res, 2016.
[4] Emmanuel Boutet, Damien Lieberherr, Michael Tognolli, Michel Schneider, and Amos Bairoch. UniProtKB/Swiss-Prot, pages 89–112. Humana Press, Totowa, NJ, 2007. ISBN 978-1-59745-535-0. doi: 10.1007/978-1-59745-535-0_4.
[5] M. Steinegger and J. Söding. Mmseqs2 enables sensitive protein sequence searching for the analysis of massive datasets. Nat Biotechnol, 35:1026–1028, 2017a. doi: https://doi.org/10.1038/nbt.3988.
[6] Mihaly Varadi, Stephen Anyango, Mandar Deshpande, Sreenath Nair, Cindy Natassia, Galabina Yordanova, David Yuan, Oana Stroe, Gemma Wood, Agata Laydon, Augustin Žídek, Tim Green, Kathryn Tunyasuvunakool, Stig Petersen, John Jumper, Ellen Clancy, Richard Green, Ankur Vora, Mira Lutfi, Michael Figurnov, Andrew Cowie, Nicole Hobbs, Pushmeet Kohli, Gerard Kleywegt, Ewan Birney, Demis Hassabis, and Sameer Velankar. AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic Acids Research, 50(D1):D439–D444, 11 2021. ISSN 0305-1048. doi: 10.1093/nar/gkab1061.
[7] R. Krivák and D. Hoksza. P2rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. J Cheminform, 10, 2018. doi: https://doi.org/10.1186/s13321-018-0285-8.
[8] https://www.clinicalgenome.org/data-sharing/clinvar/



<br>

## Main Ideas


- [**ImageBind**](https://arxiv.org/abs/2305.05665)
- [**CLIP**](https://arxiv.org/abs/2103.00020)

DownStream Tasks:

- [**SaProt**](https://www.biorxiv.org/content/10.1101/2023.10.01.)
<br>

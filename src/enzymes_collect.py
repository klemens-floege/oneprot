from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from functools import partial

import h5py
import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score, f1_score

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.utils_structure import protein_to_graph

ENZYME_DIR = Path("/p/project/hai_oneprot/sommer3/oneprot/hai_oneprot/TopEnzyme")


def identity_map(embedding, cfg):
    return embedding

def threshold_map(embedding, cfg):
    dev = embedding.device
    return torch.where(embedding > cfg.bit_threshold, torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev))

function_map = {
    "identity": identity_map,
    "threshold": threshold_map,
}


class EnyzmeDataset(Dataset):
    def __init__(self, partition, csv_file="Combined826_hier_", h5file="all_enzymes.h5", seq_tokenizer="facebook/esm2_t33_650M_UR50D"):
        self.partition = partition
        self.csv_path = ENZYME_DIR / (csv_file + partition + ".csv")
        self.h5_path = ENZYME_DIR / h5file
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.fasta_file = ENZYME_DIR / f"{partition}.fasta"

        with open(self.fasta_file, "r") as f:
            self.fasta_strings = f.readlines()

        self.csv_data = pd.read_csv(self.csv_path)
        self.h5_file = h5py.File(self.h5_path, 'r')

        self.fasta_strings = [line.strip() for line in self.fasta_strings]
        self.fasta_strings = [line for line in self.fasta_strings if line]
        self.fasta_strings = [line for line in self.fasta_strings if not line.startswith(">")]
        assert len(self.fasta_strings) == len(self.csv_data)

    def __len__(self):
        return len(self.csv_data)
        
    def __getitem__(self, idx):       
        return idx

    def collate_fn(self, idxs):
        sequences, structures, targets = [], [], []

        for idx in idxs:
            try:
                # load data identifier and function target
                enzyme_id = self.csv_data.iloc[idx]["enzyme_name"]
                target = self.csv_data.iloc[idx]["hierarchical"]
                # load sequence from h5 file
                sequence = self.fasta_strings[idx]
                # load structure from h5 file with seperate helper function
                structure = protein_to_graph(enzyme_id, self.h5_path, 'pdb' , 'all')
                sequences.append(sequence)
                structures.append(structure)
                targets.append(target)
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue
        
        # concatenate batch into one graph 
        batch_struct = Batch.from_data_list(structures)
        # tokenize sequence for final output
        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids   
        targets = np.array(targets)

        return sequence_input.long(), batch_struct, targets
    

def collect_all_embeddings(model, dataset, cfg, partition, output_dir):
    dataloader = DataLoader(dataset, batch_size=cfg.encode_batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn)
    trainer = pl.Trainer(
        devices=4, 
        accelerator='gpu', 
        limit_predict_batches=cfg.batch_limit,
        num_nodes=1, #num_nodes
        strategy='ddp',
    )  

    model.current_partition = partition
    model.output_dir = output_dir
    func = partial(function_map[cfg.transform_func], cfg=cfg)
    model.transform_func = func

    trainer.predict(model, dataloader)


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    model = hydra.utils.instantiate(cfg.model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
        model.cuda()
    else:
        model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"])
    model.eval()

    # instantiate dataset and collect embeddings for training structures
    for partition in cfg.evaluate_on:
        train_dataset = EnyzmeDataset(partition)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        collect_all_embeddings(model, train_dataset, cfg, partition, output_dir)


@hydra.main(version_base="1.3", config_path="../configs", config_name="downstream.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    evaluate(cfg)


if __name__ == "__main__":
    main()

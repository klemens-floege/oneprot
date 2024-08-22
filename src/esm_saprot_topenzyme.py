from typing import Tuple
from pathlib import Path
import tqdm

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
from transformers import EsmTokenizer, EsmModel

from sklearn.metrics import accuracy_score, f1_score

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.utils_structure import protein_to_graph

SA_DIR = Path("/p/project/hai_oneprot/floege1/oneprot-2/hai_oneprot/for_klemens/")


def identity_map(embedding, cfg):
    return embedding

def threshold_map(embedding, cfg):
    dev = embedding.device
    return torch.where(embedding > float(cfg.transform_func.split("_")[1]), torch.tensor(1.0).to(dev), torch.tensor(0.0).to(dev))

function_map = {
    "identity": identity_map,
    "threshold": threshold_map,
}


class GeneralDataset(Dataset):
    def __init__(self, partition, task_name, seq_tokenizer="facebook/esm2_t33_650M_UR50D"):
        self.partition = partition
        self.csv_path = SA_DIR / "csv" / (task_name + "_" + partition + ".csv")
        self.h5_path = SA_DIR / "h5" / (task_name.split("_")[0] + ".h5")
        self.seq_tokenizer = EsmTokenizer.from_pretrained(seq_tokenizer)
        self.task_name = task_name

        self.csv_data = pd.read_csv(self.csv_path)
        # self.csv_data = self.csv_data[:22453]
        self.h5_file = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.csv_data)
        
    def __getitem__(self, idx):       
        return idx

    # def collate_fn(self, idxs):
    #     sequences, targets = [], []

    #     for idx in idxs:
    #         try:
    #             # load data identifier and function target
    #             enzyme_id = self.csv_data.iloc[idx]["name"]
    #             target = self.csv_data.iloc[idx]["label"]
    #             # load sequence from h5 file
    #             with h5py.File(self.h5_path, 'r') as file:
    #                 sequence = file[enzyme_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                
    #             sequences.append(sequence)
    #             targets.append(target)
    #         except Exception as e:
    #             print(f"Error at index {idx}: {e}")
    #             continue
    
     
        
        
    #     # tokenize sequence for final output
    #     sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt")  
    #     targets = np.array(targets)
    #     return sequence_input, targets
    def collate_fn(self, idxs):
        sequences, targets = [], []

        for idx in idxs:
            try:
                # Load data identifier and function target
                enzyme_id = self.csv_data.iloc[idx]["name"]
                target = self.csv_data.iloc[idx]["label"]
            
                # Load sequence from h5 file
                with h5py.File(self.h5_path, 'r') as file:
                    sequence = []
                
                    # Access the structure group for the given enzyme ID
                    structure_group = file[enzyme_id]['structure']['0']
                
                    # Iterate through all available chains
                    for chain in structure_group.keys():
                        try:
                            # Access the sequence for the current chain and extend the sequence list
                            chain_sequence = structure_group[chain]['residues']['seq1'][()].decode('utf-8')
                            sequence.extend(chain_sequence)
                        except KeyError:
                            # Handle the case where the chain or sequence is missing
                            print(f"Chain {chain} not found or no sequence available for enzyme ID {enzyme_id}. Skipping this chain.")
                            continue
                
                    # Concatenate all sequences into a single string
                    sequence = ''.join(sequence)
            
                # Check if the sequence is empty
                if not sequence:
                    print(f"Empty sequence for enzyme_id {enzyme_id}. Skipping entry.")
                    continue
            
                sequences.append(sequence)
                targets.append(target)
        
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue

        if not sequences:
            raise ValueError("No valid sequences to process.")
            
        # tokenize sequence for final output
        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt")  
        targets = np.array(targets)
        return sequence_input, targets

 
            
        
    def collate_humanppi_fn(self, idxs):
        sequences_1, sequences_2, targets = [], [], []

        for idx in idxs:
            try:
                # load data identifiers and function target
                protein_id_1 = self.csv_data.iloc[idx]["name_1"]
                protein_id_2 = self.csv_data.iloc[idx]["name_2"]
                target = self.csv_data.iloc[idx]["label"]
                
                # load sequences from h5 file
                with h5py.File(self.h5_path, 'r') as file:
                    sequence_1 = file[protein_id_1]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                    sequence_2 = file[protein_id_2]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')


                sequences_1.append(sequence_1)
                sequences_2.append(sequence_2)
                targets.append(target)
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue

        # tokenize sequences for final output
        sequence_input_1 = self.seq_tokenizer(sequences_1, max_length=1024, padding=True, truncation=True, return_tensors="pt")   
        sequence_input_2 = self.seq_tokenizer(sequences_2, max_length=1024, padding=True, truncation=True, return_tensors="pt")
        targets = np.array(targets)
        return sequence_input_1, sequence_input_2, targets


# def collect_all_embeddings(model, dataset, cfg, partition, output_dir, device):
#     if cfg.task_name in ["HumanPPI"]:
#         dataloader = DataLoader(dataset, batch_size=cfg.encode_batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_humanppi_fn)
#     else: 
#         dataloader = DataLoader(dataset, batch_size=cfg.encode_batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn)

#     # func = partial(function_map[cfg.transform_func.split("_")[0]], cfg=cfg)
#     batch_idx = 0
#     for batch in tqdm.tqdm(dataloader):
#         seq_inputs, targets = batch
#         inputs = {key: value.to(device) for key, value in seq_inputs.items()}
#         with torch.no_grad():
#             outputs = model(**inputs)
#             # Taking the mean of the token embeddings to get a single vector per sequence
#             seq_embeddings = outputs.last_hidden_state.mean(dim=1)
#             seq = function_map[cfg.transform_func](seq_embeddings, cfg)
#             gpu = torch.cuda.current_device()
#             local_path = output_dir / partition / f"{str(gpu)}"
#             if not local_path.exists():
#                 local_path.mkdir(parents=True)
#             np.save(local_path / f"{batch_idx}_seq.npy", seq.detach().cpu().numpy())
#             np.save(local_path / f"{batch_idx}_target.npy", targets)
#             batch_idx += 1

def collect_all_embeddings(model, dataset, cfg, partition, output_dir, device):
    if cfg.task_name in ["HumanPPI"]:
        dataloader = DataLoader(dataset, batch_size=cfg.encode_batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_humanppi_fn)
    else: 
        dataloader = DataLoader(dataset, batch_size=cfg.encode_batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn)

    batch_idx = 0
    for batch in tqdm.tqdm(dataloader):
        if batch is None:
            print(f"Batch {batch_idx} is empty. Skipping.")
            batch_idx += 1
            continue
        
        seq_inputs, targets = batch
        inputs = {key: value.to(device) for key, value in seq_inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            seq_embeddings = outputs.last_hidden_state.mean(dim=1)
            seq = function_map[cfg.transform_func](seq_embeddings, cfg)
            gpu = torch.cuda.current_device()
            local_path = output_dir / f"{partition}_01" / f"{str(gpu)}"
            if not local_path.exists():
                local_path.mkdir(parents=True)
            np.save(local_path / f"{batch_idx}_seq.npy", seq.detach().cpu().numpy())
            np.save(local_path / f"{batch_idx}_target.npy", targets)
            
            torch.cuda.empty_cache()
            
            batch_idx += 1


def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    # Initialize the model from the huggingface repository
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Use multiple GPUs if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)


    # instantiate dataset and collect embeddings for training structures
    for partition in cfg.evaluate_on:
        print("Taskname: ", cfg.task_name)
        train_dataset = GeneralDataset(partition, cfg.task_name)
        output_dir = Path(cfg.output_dir)
        output_dir = output_dir / cfg.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        collect_all_embeddings(model, train_dataset, cfg, partition, output_dir,device)


@hydra.main(version_base="1.3", config_path="../configs", config_name="esm_saprot_topenzyme.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    evaluate(cfg)


if __name__ == "__main__":
    main()

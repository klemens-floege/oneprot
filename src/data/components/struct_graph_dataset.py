import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch_geometric.data import Batch, Data
from typing import List, Tuple, Any
from src.data.components.struct_graph_utils import protein_to_graph
import os



class StructDataset(Dataset):
    def __init__(self, data_dir: str, split: str, seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D", 
                 use_struct_mask: bool = False, use_struct_coord_noise: bool = False, 
                 use_struct_deform: bool = False, pockets: bool = False, seqsim: str = '50ss'):
        self.id_list = []
        self.pockets = pockets
        self.split = split
        self.h5_file = f'{data_dir}/{"pockets_100_residues" if pockets else "merged"}.h5'

        self.use_struct_mask = use_struct_mask
        self.use_struct_coord_noise = use_struct_coord_noise
        self.use_struct_deform = use_struct_deform

        csv_file = f'{data_dir}/{seqsim}/{split}_{"pocket" if pockets else "seqstruc"}.csv'
        try:
            with open(csv_file, 'r') as file:
                self.id_list = [line.split(',')[0].strip() for line in file]
        except FileNotFoundError:
            print(f"File not found: {csv_file}")

        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)

    def __len__(self) -> int:
        return len(self.id_list)
        
    def __getitem__(self, idx: int) -> str:       
        return self.id_list[idx]

    def collate_fn(self, seq_ids: List[str], return_raw_sequences: bool = False) -> Tuple[torch.Tensor, Any]:
        sequences, structures = [], []
        for seq_id in seq_ids:
            try:
                with h5py.File(self.h5_file, 'r') as file:
                    sequence = file[seq_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                    sequences.append(sequence)
                structures.append(protein_to_graph(seq_id, self.h5_file, 'non_pdb', 'A', pockets=self.pockets))
            except KeyError:
                print(f"KeyError: {seq_id} not found in {self.h5_file}")

        if return_raw_sequences:
            return sequences
        
        batch_struct = Batch.from_data_list(structures)

        if self.use_struct_mask and self.split=='train':
            mask_aatype = np.random.uniform(0, 1)
            mask_indice = torch.tensor(np.random.choice(batch_struct.num_nodes, int(batch_struct.num_nodes * mask_aatype), replace=False))
            batch_struct.x[:, 0][mask_indice] = 20
        if self.use_struct_coord_noise and self.split=='train':
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch_struct.pos.shape), min=-0.3, max=0.3)
            batch_struct.pos += gaussian_noise
        if self.use_struct_deform and self.split=='train':
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch_struct.pos *= deform

        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids   
        return sequence_input.long(), batch_struct
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple

class TextDataset(Dataset):
    def __init__(self, data_dir: str, split: str, max_length: int = 1024, text_max_length: int = 512, text_tokenizer: str = "allenai/scibert_scivocab_uncased", 
                 seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D"):
        
        
        self.text_max_length = text_max_length
        self.max_length = max_length
        self.h5_file = f'{data_dir}/seqstruc.h5'
        self.split = split
        csv_file = f'{data_dir}/{split}_text.csv'

        
        try:
            self.df = pd.read_csv(csv_file, header=None)
        except FileNotFoundError:
            print(f"File not found: {csv_file}")
            self.df = pd.DataFrame()
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
       
    def __len__(self) -> int:
        
        if self.split == "train":
            return self.df.shape[0]
        return 1000
        
    def __getitem__(self, idx: int) -> str:
        return self.df[0].iloc[idx]

    def collate_fn(self, data: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        texts = []
        for seq_id in data:
            ind = self.df[self.df[0] == seq_id].index.tolist()[0]
            try:
                with h5py.File(self.h5_file, 'r') as file:
                    sequence = file[seq_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                    sequences.append(sequence)
                texts.append(self.df[1].iloc[ind])
            except KeyError:
                print(f"KeyError: {seq_id} not found in {self.h5_file}")
            
        sequence_input = self.seq_tokenizer(sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids   
        text_input = self.text_tokenizer(texts, max_length=self.text_max_length, padding=True, truncation=True, return_tensors="pt").input_ids   
        modality = "text"
        return sequence_input.long(), text_input.long(), modality, sequences
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmTokenizer
import pandas as pd
from typing import List, Tuple
import h5py

class StructTokenDataset(Dataset):
    def __init__(self, data_dir: str, filename: str, split: str, max_length: int = 1024, 
                 seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D", remove_hash=True):
        """
        Initialize the Structure Transformation Dataset.
        
        Args:
            data_dir (str): Directory containing the data.
            split (str): Data split ('train', 'val', or 'test').
            struct_tokenizer (str): Structure tokenizer model name.
            seq_tokenizer (str): Sequence tokenizer model name.
            seqsim (str): Sequence similarity threshold.
        """
        self.split = split
        self.remove_hash = remove_hash
        self.max_length = max_length
        txt_file = f'{data_dir}/{self.split}_saprot.txt'
  
        try:
            with open(txt_file, 'r') as file:
                self.ids = [line.strip() for line in file]
        except FileNotFoundError:
            print(f"File not found: {txt_file}")

        new_tokens = ['p', 'y', 'n', 'w', 'r', 'q', 'h', 'g', 'd', 'l', 'v', 't', 'm', 'f', 's', 'a', 'e', 'i', 'k', 'c','#']
        self.struct_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.struct_tokenizer.add_tokens(new_tokens)    
        
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)

        self.filename = filename
       
    def __len__(self) -> int:
        
        if self.split == "train":
            return len(self.ids)
        return 1000
    def __getitem__(self, idx: int) -> str:
        return self.ids[idx]

    def collate_fn(self, data: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for the Structure Transformation Dataset.
        
        Args:
            data (List[str]): List of sequence IDs.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tokenized sequence input and structure input.
        """
        
        sequences = []
        structs = []
        # Open the HDF5 file
        with h5py.File(self.filename, 'r') as h5_file:
    
            for seq_id in data:
   
                if seq_id in h5_file:
                    strucseq = h5_file[seq_id]['strucseq'][()].decode('utf-8')
                    sequence = ''.join(strucseq[i] for i in range(0, len(strucseq), 2))  # Odd-indexed tokens (sequence)
                    sequence = sequence.replace("#", "")

                    structure_seq = ''.join(strucseq[i] for i in range(1, len(strucseq), 2))  # Even-indexed tokens (structure_seq)
                    
                    # Remove "#" if the flag is set
                    if self.remove_hash:
                        
                        structure_seq = structure_seq.replace("#", "")

                    structs.append(structure_seq)
                    sequences.append(sequence)
         
        sequence_input = self.seq_tokenizer(sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids   
        struct_input = self.struct_tokenizer(structs, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids   
        modality = "struct_token"
        return sequence_input.long(), struct_input.long(), modality, sequences
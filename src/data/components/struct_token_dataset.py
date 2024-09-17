import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmTokenizer
import pandas as pd
from typing import List, Tuple

class StructTokenDataset(Dataset):
    def __init__(self, data_dir: str, split: str, 
                 struct_tokenizer: str = "westlake-repl/SaProt_650M_AF2",
                 seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D", seqsim: str = '50ss'):
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
        csv_file = f'{data_dir}/output_saprot_{self.split}.csv'
        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"File not found: {csv_file}")
            self.df = pd.DataFrame()
        self.struct_tokenizer = EsmTokenizer.from_pretrained(struct_tokenizer)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
       
    def __len__(self) -> int:
        return self.df.shape[0]
        
    def __getitem__(self, idx: int) -> str:
        return self.df['id'].iloc[idx]

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
        
        for seq_id in data:
            ind = self.df[self.df['id'] == seq_id].index.tolist()[0]
            structs.append(self.df['seqstruc'].iloc[ind])
            sequences.append(self.df['seq'].iloc[ind])
            
        sequence_input = self.seq_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids   
        struct_input = self.struct_tokenizer(structs, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids   
        return sequence_input.long(), struct_input.long()
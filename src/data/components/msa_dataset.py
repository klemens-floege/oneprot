import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import esm
from typing import List, Tuple


class MSADataset(Dataset):
    def __init__(self, data_dir: str, split: str, max_length: int = 1024, 
                 msa_depth: int = 100, seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D",
                 seqsim: str = '50ss'):
        """
        Initialize the MSA Dataset.
        
        Args:
            data_dir (str): Directory containing the data.
            split (str): Data split ('train', 'val', or 'test').
            max_length (int): Maximum sequence length.
            msa_depth (int): Depth of MSA.
            seq_tokenizer (str): Sequence tokenizer model name.
            seqsim (str): Sequence similarity threshold.
        """
        filename = f"{data_dir}/{seqsim}/{split}_msa.csv"
        self.msa_files = filter_and_create_msa_file_list(filename)
        _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_padding_idx = 1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1022)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.max_length = max_length
        self.msa_depth = msa_depth
        self.split = split
 
    def __len__(self) -> int:
        return len(self.msa_files)
       
    def __getitem__(self, idx: int) -> str:
        return self.msa_files[idx]

    def collate_fn(self, msa_files: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        msas = []
        for msa_file in msa_files:
            msa_data = read_msa(msa_file)
            msa_data = greedy_select(msa_data, num_seqs=self.msa_depth)
            sequence = msa_data[0][1]
            sequences.append(sequence)
            msas.append(msa_data)

        _, _, msa_input = self.msa_transformer_batch_converter(msas)
        msa_input = msa_input[:, :, :self.max_length]
        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids   
        
        return sequence_input.long(), torch.as_tensor(msa_input).long()
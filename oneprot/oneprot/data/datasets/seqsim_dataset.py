import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import os
import pandas as pd
from typing import List, Tuple, Dict

class SequenceSimDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D",
        max_length: int = 1024,
        modality: str = "combined_seqsim_msa"
    ):
        """
        Initialize the SequenceSimDataset.

        Args:
            data_dir (str): Directory containing the data files.
            split (str): Data split ('train', 'val', or 'test').
            seq_tokenizer (str): Name of the sequence tokenizer model.
            max_length (int): Maximum sequence length for tokenization.
            modality (str): Modality identifier for the dataset.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.modality = modality

        self._load_data()

    def _load_data(self):
        """Load sequence IDs, mutation dictionaries, and MSA data."""
        with open(os.path.join(self.data_dir, f'{self.split}_seqsim.txt'), 'r') as f:
            self.sequence_ids = [line.strip() for line in f]
        
        self.benign_mutations = self._load_json('clinvar_full_benign_mutations.json')
        self.pathogenic_mutations = self._load_json('clinvar_full_pathogenic_mutations.json')

        # Load MSA data from CSV file
        msa_filename = f"{self.data_dir}/{self.split}_msa_seqsim.csv"
        self.msa_data = pd.read_csv(msa_filename)

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file."""
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self.split == "train":
            return len(self.msa_data)
        return 1000

    def __getitem__(self, idx: int) -> Tuple[str, pd.Series]:
        """Get a single item from the dataset."""
        seq_id = self.sequence_ids[idx % len(self.sequence_ids)]
        msa_row = self.msa_data.iloc[idx]
        return seq_id, msa_row

    @staticmethod
    def _apply_mutation(sequence: str, mutation: str) -> str:
        """Apply a specific mutation to the sequence."""
        letter1, position, letter2 = mutation[0], int(mutation[1:-1]), mutation[-1]
        position -= 1  # Adjust for 0-based indexing
        assert sequence[position] == letter1, f"Mutation mismatch: expected {letter1} at position {position}, found {sequence[position]}"
        return sequence[:position] + letter2 + sequence[position+1:]

    def _get_msa_sequence(self, msa_row: pd.Series) -> Tuple[str, str]:
        """Get the original and aligned MSA sequence from the MSA data."""
        original_seq = msa_row['req_seq']
        aligned_seq = msa_row['aligned_seq']
        return original_seq, aligned_seq

    def collate_fn(self, batch: List[Tuple[str, pd.Series]]) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Collate function for DataLoader."""
        list1 = []  # Will contain original_msa, seq_id, pathogenic1
        list2 = []  # Will contain aligned_seq, benign, pathogenic2

        for seq_id, msa_row in batch:
            # Get original and aligned sequences
            original_msa, aligned_seq = self._get_msa_sequence(msa_row)
            
            # Add original_msa to list1 and aligned_seq to list2
            list1.append(original_msa)
            list2.append(aligned_seq)

            # Add seq_id to list1
            list1.append(seq_id)

            # Benign mutation
            while True:
                benign_mutation = random.choice(self.benign_mutations[seq_id])
                try:
                    benign = self._apply_mutation(seq_id, benign_mutation)
                    list2.append(benign)
                    break
                except AssertionError:
                    continue

            # Pathogenic mutations
            pathogenic_mutations = []
            while len(pathogenic_mutations) < 2:
                mutation = random.choice(self.pathogenic_mutations[seq_id])
                try:
                    pathogenic = self._apply_mutation(seq_id, mutation)
                    pathogenic_mutations.append(pathogenic)
                except AssertionError:
                    continue

            # Add first pathogenic mutation to list1
            list1.append(pathogenic_mutations[0])

            # Add second pathogenic mutation to list2
            list2.append(pathogenic_mutations[1])

        # Tokenize sequences
        sequence_input1 = self.seq_tokenizer(list1, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        sequence_input2 = self.seq_tokenizer(list2, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        modality = "seqsim"
        return sequence_input1.long(), sequence_input2.long(), modality, sequence_input1
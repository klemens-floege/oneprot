import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import os
from typing import List, Tuple, Dict
from src.data.utils.msa_utils import read_msa, filter_and_create_msa_file_list, greedy_select

class SequenceSimDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D",
        max_length: int = 1024,
        msa_depth: int = 100,
        modality: str = "combined_seqsim_msa"
    ):
        """
        Initialize the CombinedSeqSimMsaDataset.

        Args:
            data_dir (str): Directory containing the data files.
            split (str): Data split ('train', 'val', or 'test').
            seq_tokenizer (str): Name of the sequence tokenizer model.
            max_length (int): Maximum sequence length for tokenization.
            msa_depth (int): Depth of MSA sequences.
            modality (str): Modality identifier for the dataset.
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.msa_depth = msa_depth
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.modality = modality

        self._load_data()

    def _load_data(self):
        """Load sequence IDs, mutation dictionaries, and MSA files."""
        with open(os.path.join(self.data_dir, f'{self.split}_seqsim.txt'), 'r') as f:
            self.sequence_ids = [line.strip() for line in f]
        
        self.benign_mutations = self._load_json('clinvar_full_benign_mutations_clean.json')
        self.pathogenic_mutations = self._load_json('clinvar_full_pathogenic_mutations_clean.json')

        msa_filename = f"{self.data_dir}/{self.split}_msa.csv"
        self.msa_files = filter_and_create_msa_file_list(msa_filename)

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file."""
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self.split == "train":
            return len(self.msa_files)
        return 1000

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get a single item from the dataset."""
        seq_id = self.sequence_ids[idx % len(self.sequence_ids)]
        msa_file = self.msa_files[idx]
        return seq_id, msa_file

    @staticmethod
    def _apply_mutation(sequence: str, mutation: str) -> str:
        """Apply a specific mutation to the sequence."""
        letter1, position, letter2 = mutation[0], int(mutation[1:-1]), mutation[-1]
        position -= 1  # Adjust for 0-based indexing
        assert sequence[position] == letter1, f"Mutation mismatch: expected {letter1} at position {position}, found {sequence[position]}"
        return sequence[:position] + letter2 + sequence[position+1:]

    def _get_msa_sequence(self, msa_file: str) -> Tuple[str, str]:
        """Get the original and a random MSA sequence from the MSA file."""
        msa_data = read_msa(msa_file)
        random_mode = random.choice(['max', 'min'])
        msa_data = greedy_select(msa_data, num_seqs=self.msa_depth, mode=random_mode)
        original_seq = msa_data[0][1]
        random_ind = random.randint(1, len(msa_data) - 1)
        msa_seq = msa_data[random_ind][1]
        return original_seq, msa_seq

    def collate_fn(self, batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Collate function for DataLoader."""
        list1 = []  # Will contain original_msa, seq_id, pathogenic1
        list2 = []  # Will contain msa_seq, benign, pathogenic2

        for seq_id, msa_file in batch:
            # Get original and MSA sequences
            original_msa, msa_seq = self._get_msa_sequence(msa_file)
            
            # Add original_msa to list1 and msa_seq to list2
            list1.append(original_msa)
            list2.append(msa_seq)

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
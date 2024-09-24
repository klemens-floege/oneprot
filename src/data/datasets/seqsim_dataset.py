import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import os
from typing import List, Tuple

class SequenceSimDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        seq_tokenizer: str = "facebook/esm2_t33_650M_UR50D",
        max_length: int = 1024
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)

        # Load sequence IDs
        with open(os.path.join(data_dir, f'{split}_seqsim.txt'), 'r') as f:
            self.sequence_ids = [line.strip() for line in f]
        
        # Load mutation dictionaries
        with open(os.path.join(data_dir, 'clinvar_full_benign_mutations_clean.json'), 'r') as f:
            self.benign_mutations = json.load(f)

        with open(os.path.join(data_dir, 'clinvar_full_pathogenic_mutations_clean.json'), 'r') as f:
            self.pathogenic_mutations = json.load(f)


    def __len__(self):
        
        if self.split == "train":
             return len(self.sequence_ids)
        else:
            return 250
       
    def __getitem__(self, idx: int) -> str:
        return self.sequence_ids[idx]

    def apply_mutation(self, sequence, mutation):
        letter1, position, letter2 = mutation[0], int(mutation[1:-1]), mutation[-1]
        position -= 1  # Adjust for 0-based indexing
        assert sequence[position] == letter1, f"Mutation mismatch: expected {letter1} at position {position}, found {sequence[position]}"
        return sequence[:position] + letter2 + sequence[position+1:]

    def collate_fn(self, seq_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_sequences = []
        mutated_sequences = []
      
        for seq_id in seq_ids:
  
            original_sequences.append(seq_id)

            # Apply benign mutation
            while True:
                try:
                    benign_mutation = random.choice(self.benign_mutations[seq_id])
                    benign_sequence = self.apply_mutation(seq_id, benign_mutation)
                    mutated_sequences.append(benign_sequence)
                    break  # Exit the loop if mutation is successful
                except Exception as e:
                    print(f"Error applying mutation: {e}. Sampling a different mutation.")

            while True:
                try:
                    # Apply pathogenic mutation
                    pathogenic_mutation = random.choice(self.pathogenic_mutations[seq_id])
                    pathogenic_sequence = self.apply_mutation(seq_id, pathogenic_mutation)
                    original_sequences.append(pathogenic_sequence)
                    break
                except Exception as e:
                    print(f"Error applying mutation: {e}. Sampling a different mutation.")
            
            while True:
                try:
                    # Apply pathogenic mutation
                    pathogenic_mutation = random.choice(self.pathogenic_mutations[seq_id])
                    pathogenic_sequence = self.apply_mutation(seq_id, pathogenic_mutation)
                    mutated_sequences.append(pathogenic_sequence)
                    break
                except Exception as e:
                    print(f"Error applying mutation: {e}. Sampling a different mutation.")
                    
        # Tokenize sequences
        original_input = self.seq_tokenizer(original_sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        mutated_input = self.seq_tokenizer(mutated_sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        modality = "seqsim"
        return original_input, mutated_input, modality, original_sequences
from transformers import EsmTokenizer, EsmModel
import tqdm
import torch
import numpy as np
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

# Function to load sequences from a FASTA file
def load_sequences_from_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

class ProteinDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch):
    return tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

# Function to tokenize the sequences and get the embeddings
def get_protein_embeddings(sequences, batch_size=4):
    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    embeddings = []
    for inputs in tqdm.tqdm(dataloader):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Taking the mean of the token embeddings to get a single vector per sequence
            seq_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(seq_embeddings)
    return np.vstack(embeddings)

# Initialize the tokenizer and model from the huggingface repository
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Use multiple GPUs if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

# Load sequences from FASTA files
train_sequences = load_sequences_from_fasta('/p/project/hai_oneprot/floege1/oneprot-2/hai_oneprot/TopEnzyme/train.fasta')
val_sequences = load_sequences_from_fasta('/p/project/hai_oneprot/floege1/oneprot-2/hai_oneprot/TopEnzyme/val.fasta')
test_sequences = load_sequences_from_fasta('/p/project/hai_oneprot/floege1/oneprot-2/hai_oneprot/TopEnzyme/test.fasta')

# Extract embeddings for each set
train_embeddings = get_protein_embeddings(train_sequences)
val_embeddings = get_protein_embeddings(val_sequences)
test_embeddings = get_protein_embeddings(test_sequences)

# Save embeddings as numpy arrays
np.save("train_embeddings.npy", train_embeddings)
np.save("val_embeddings.npy", val_embeddings)
np.save("test_embeddings.npy", test_embeddings)

print("Embeddings saved as train_embeddings.npy, val_embeddings.npy, and test_embeddings.npy")
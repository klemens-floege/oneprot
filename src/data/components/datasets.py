import pickle
import json
from src.data.components.utils_msa import filter_and_create_msa_file_list, greedy_select, read_msa
from src.data.components.utils_structure import protein_to_graph
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import esm
import numpy as np
import torch
import h5py
import pandas as pd

class MSADataset(Dataset):
    def __init__(self, data_dir="/p/scratch/hai_oneprot/openfoldh5s", split='train', max_length=1024, msa_depth=100, seq_tokenizer="facebook/esm2_t33_650M_UR50D"):
        
        filename = f"{data_dir}/msa_{split}_files.txt"
        self.msa_files = filter_and_create_msa_file_list(filename)
        _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_padding_idx =1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1024)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.max_length = max_length
        self.msa_depth = msa_depth
         # can change this to pass more/fewer sequences
 
    def __len__(self):
        #return 2000
        return len(self.msa_files)
       
    def __getitem__(self, idx):
     
        return idx


    def collate_fn(self, data):
        """
        data: is a list of tuples with (example, label, length)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        sequences = []
        msas = []
        for i in range(len(data)):

            msa_data = read_msa(self.msa_files[data[i]])
            msa_data = greedy_select(msa_data, num_seqs=self.msa_depth)
        
            sequence = msa_data[0][1]
            
            sequences.append(sequence)
            msas.append(msa_data)

        _, _, msa_input = self.msa_transformer_batch_converter(msas)

        sequence_input = self.seq_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids   
        
        return sequence_input.long(), torch.as_tensor(msa_input).long()

class StructDataset(Dataset):

    def __init__(self, data_dir = '/p/scratch/hai_oneprot/openfoldh5s', split="train", seq_tokenizer="facebook/esm2_t33_650M_UR50D", use_struct_mask=False, use_struct_coord_noise=False, use_struct_deform=False):
         
        
        self.id_list = []
        self.split = split
        self.h5_file =  f'{data_dir}/merged.h5'
        self.use_struct_mask = use_struct_mask
        self.use_struct_coord_noise = use_struct_coord_noise
        self.use_struct_deform = use_struct_deform
        with open(f'{data_dir}/{split}_struct.txt', 'r') as file:

            for line in file:
                self.id_list.append(line.split(',')[0].strip())  

        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
    def __len__(self):
        return len(self.id_list)
        #return 2000
        
    def __getitem__(self, idx):
               
        return idx

    def collate_fn(self, data):
        """
        data: is a list of tuples with (example, label, length)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """

        seq_ids = data
        sequences = []
        structures = []
        for i in range(len(seq_ids)):
            with h5py.File(self.h5_file, 'r') as file:
                sequence = file[self.id_list[seq_ids[i]]]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                sequences.append(sequence)
            structures.append(protein_to_graph(self.id_list[seq_ids[i]], self.h5_file, 'non_pdb' , 'A'))
        
        batch_struct = Batch.from_data_list(structures)

        if self.use_struct_mask:
            # random mask node aatype
            mask_aatype = np.random.uniform(0, 1)
            mask_indice = torch.tensor(np.random.choice(batch_struct.num_nodes, int(batch_struct.num_nodes * mask_aatype), replace=False))
            batch_struct.x[:, 0][mask_indice] = 20
        if self.use_struct_coord_noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch_struct.coords_ca.shape), min=-0.3, max=0.3)
            batch_struct.coords_ca += gaussian_noise
            batch_struct.coords_n += gaussian_noise
            batch_struct.coords_c += gaussian_noise
        if self.use_struct_deform:
            # Anisotropic scale
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch_struct.coords_ca *= deform
            batch_struct.coords_n *= deform
            batch_struct.coords_c *= deform

        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids   
       
        return sequence_input.long(), batch_struct

'''
class TextDataset(Dataset):
    def __init__(self, data_dir = '/p/scratch/hai_oneprot/openfoldh5s/' , split="train", text_tokenizer="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", seq_tokenizer="facebook/esm2_t33_650M_UR50D"):
      
        self.uniprot_text = {}
        self.uniprot_protein = {}

        with open(f'{data_dir}text_{split}.txt', 'r') as file:
            
            self.id_list = file.read().splitlines()

        with open(f'{data_dir}text_sequence.txt', 'r') as file:
            
            lines = file.read().splitlines()

        # Iterate over the lines two by two (ID, description)
        for i in range(0, len(lines), 2):
            if i+1 < len(lines):
                self.uniprot_text[lines[i]] = lines[i+1]

        with open(f'{data_dir}protein_sequence.txt', 'r') as file:
            
            lines = file.read().splitlines()

        # Iterate over the lines two by two (ID, description)
        for i in range(0, len(lines), 2):
            if i+1 < len(lines):
                self.uniprot_protein[lines[i]] = lines[i+1]


        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
       
 
    def __len__(self):
        return len(self.id_list)
        #return 1000

    def __getitem__(self, idx):
        
        sequence = self.uniprot_protein[self.id_list[idx]]
        text = self.uniprot_text[self.id_list[idx]]

        text_input = self.text_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids
        sequence_input = self.seq_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids 
  
        return torch.squeeze(sequence_input), torch.squeeze(text_input)


def text_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    sequences, texts = zip(*data)

    max_length = 0
    for sequence in sequences:
        if sequence.shape[0]>max_length:
            max_length = sequence.shape[0]

    sequence_input = torch.ones((len(sequences), max_length))
    for idx, sequence in enumerate(sequences):
      
        sequence_input[idx,:sequence.shape[0]] = sequence
    
    max_length = 0
    for text in texts:
        if text.shape[0]>max_length:
            max_length = text.shape[0]

    text_input = torch.ones((len(texts), max_length))
    for idx, text in enumerate(texts):
            text_input[idx,:text.shape[0]] = text


    return sequence_input.long(), text_input.long()




class GODataset(Dataset):
    def __init__(self, go_filepath="/p/scratch/hai_oneprot/go_data_dict.pkl", seq_tokenizer="facebook/esm2_t33_650M_UR50D"):
                
        with open(go_filepath, 'rb') as file:
            self.loaded_dict = pickle.load(file)
        
        #go_embs = np.load('/p/project/hai_oneprot/merdivan1/embeddings.npz', allow_pickle=True)['embds'].item()
        #self.go_emb_token = {}
        #ind = 2
        #for go_emb_key, _ in go_embs.items():
        #    self.go_emb_token[int(go_emb_key.replace("GO:",""))] = ind
        #    ind = ind + 1   
        
        with open('/p/scratch/hai_oneprot/go_emb_token.pkl', 'rb') as file:
            self.go_emb_token = pickle.load(file)
        
        self.ids = list(self.loaded_dict.keys())
         # can change this to pass more/fewer sequences
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)

    def __len__(self):
        return 10000
        #return len(self.ids)
        
    def __getitem__(self, idx):
        
        sequence = self.loaded_dict[self.ids[idx]]['sequence']
        go_terms = self.loaded_dict[self.ids[idx]]['GO']
        go_token_list = go_terms.strip().split('; ')
        
        counter = 0
        for go_term in go_token_list:
            if go_term in self.go_emb_token and counter<512:
                counter = counter+1

        go_input = torch.ones((counter+1))
        go_input[0] = 0
        counter = 0
        for go_term in go_token_list:
            if go_term in self.go_emb_token and counter<512:
                go_input[counter+1:counter+2] = torch.tensor(self.go_emb_token[go_term])
                counter = counter + 1
        
        sequence_input = self.seq_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids    

        return torch.squeeze(sequence_input), go_input


def go_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    sequences, go_terms = zip(*data)
    max_length = 0
    for sequence in sequences:
        if sequence.shape[0]>max_length:
            max_length = sequence.shape[0]

    sequence_input = torch.ones((len(sequences), max_length))
    for idx, sequence in enumerate(sequences):
        sequence_input[idx,:sequence.shape[0]] = sequence

    max_length = 0
    for go_term in go_terms:
        if go_term.shape[0]>max_length:
            max_length = go_term.shape[0]

    go_input = torch.ones((len(go_terms), max_length))
    for idx, go in enumerate(go_terms):
        go_input[idx,:go.shape[0]] = go

    return sequence_input.long(), go_input.long()
'''
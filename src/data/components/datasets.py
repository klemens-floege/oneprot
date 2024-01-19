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
    def __init__(self, data_dir="/p/scratch/hai_oneprot/openfoldh5s/", split='train', max_length=1024, msa_depth=100, sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
        
        filename = f"{data_dir}{split}_MSA_clean.txt"
        self.msa_files = filter_and_create_msa_file_list(filename)
        _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_padding_idx =1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1024)
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
        self.max_length = max_length
        self.msa_depth = msa_depth
         # can change this to pass more/fewer sequences
 
    def __len__(self):
        #return 2000
        return len(self.msa_files)
       
    def __getitem__(self, idx):
        
        msa_data = read_msa(self.msa_files[idx])
        msa_data = greedy_select(msa_data, num_seqs=self.msa_depth)
        
        sequence = msa_data[0][1]
        _, _, msa_input = self.msa_transformer_batch_converter(msa_data)

        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids    
        return torch.squeeze(sequence_input), msa_input


def msa_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    sequences, msa_datas = zip(*data)

    max_length = 0
    for sequence in sequences:
        if sequence.shape[0]>max_length:
            max_length = sequence.shape[0]

    sequence_input = torch.ones((len(sequences), max_length))
    for idx, sequence in enumerate(sequences):
        sequence_input[idx,:sequence.shape[0]] = sequence

    max_length = 0
    max_depth = 0
    for msa_data in msa_datas:
        if msa_data.shape[1]>max_depth:
            max_depth = msa_data.shape[1]
        if msa_data.shape[2]>max_length:
            max_length = msa_data.shape[2]

    msa_input = torch.ones((len(msa_datas), max_depth, min(max_length, 1024)))    
    for idx, msa_data in enumerate(msa_datas):
        msa_input[idx, :msa_data.shape[1], :min(msa_data.shape[2],1024)] = msa_data[:,:,:1024]
    
    return sequence_input.long(), msa_input.long()

class StructureDataset(Dataset):

    def __init__(self, data_dir = '/p/scratch/hai_oneprot/openfoldh5s/', split="train", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
         
        
        self.id_list = []
        self.split = split
        self.h5_file =  f'{data_dir}merged.h5'
        with open(f'{data_dir}{split}_clean_750k.txt', 'r') as file:

            for line in file:
                self.id_list.append(line.split(',')[0].strip())  

        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
    def __len__(self):
        return len(self.id_list)
        #return 2000
        
    def __getitem__(self, idx):
              
        with h5py.File(self.h5_file, 'r') as file:
            sequence = file[self.id_list[idx]]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
            
        structure_input = protein_to_graph(self.id_list[idx], self.h5_file, 'non_pdb' , 'A')
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids    

        return torch.squeeze(sequence_input), structure_input

def structure_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    sequences, structures = zip(*data)
    max_length = 0
    for sequence in sequences:
        if sequence.shape[0]>max_length:
            max_length = sequence.shape[0]

    sequence_input = torch.ones((len(sequences), max_length))
    for idx, sequence in enumerate(sequences):
      
        sequence_input[idx,:sequence.shape[0]] = sequence

    return sequence_input.long(), Batch.from_data_list(structures)


class TextDataset(Dataset):
    def __init__(self, data_dir = '/p/scratch/hai_oneprot/openfoldh5s/' , split="train", text_tokenizer="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
      
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
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
       
 
    def __len__(self):
        return len(self.id_list)
        #return 1000

    def __getitem__(self, idx):
        
        sequence = self.uniprot_protein[self.id_list[idx]]
        text = self.uniprot_text[self.id_list[idx]]

        text_input = self.text_tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids 
  
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

'''


class GODataset(Dataset):
    def __init__(self, go_filepath="/p/scratch/hai_oneprot/go_data_dict.pkl", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
                
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
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)

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
        
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids    

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
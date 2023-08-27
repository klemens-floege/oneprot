import pickle
import json
from src.data.components.utils import filter_and_create_msa_file_list, greedy_select, read_msa, protein_to_graph, etract_chain_from_pdb
from torch_geometric.data import Batch, Dataset
from transformers import AutoTokenizer
import esm
import numpy as np
import torch
sequence_model = "facebook/esm2_t12_35M_UR50D"

class MSADataset(Dataset):
    def __init__(self, msa_filepath="/p/project/hai_oneprot/merdivan1/files_list.txt", max_length=1024, msa_depth=16, sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
        
        self.msa_files = filter_and_create_msa_file_list(msa_filepath)
        
        _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_padding_idx =1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1024)
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
        self.max_length = max_length
        self.msa_depth = msa_depth
         # can change this to pass more/fewer sequences
 
    def __len__(self):
        return len(self.msa_files)


    def __getitem__(self, idx):
        
        msa_data = read_msa(self.msa_files[idx])
        msa_data = greedy_select(msa_data, num_seqs=self.msa_depth)
        
        sequence = msa_data[0][1]
        _, _, msa_tokens = self.msa_transformer_batch_converter(msa_data)
        
    
        if msa_tokens.shape[2]>1024:
            msa_tokens = msa_tokens[:,:,:1024]

        msa_input = torch.ones((self.msa_depth, 1024))
        msa_input[:msa_tokens.shape[1],:msa_tokens.shape[2]] = msa_tokens

        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding='max_length', truncation=True, return_tensors="pt")   
        sequence_input['input_ids'] = torch.squeeze(sequence_input['input_ids'])
        sequence_input['attention_mask'] = torch.squeeze(sequence_input['attention_mask'])
        return sequence_input, msa_input.long()
    

class TextDataset(Dataset):
    def __init__(self, text_filepath="/p/scratch/hai_oneprot/func_data_dict.pkl", text_tokenizer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
        
        
        with open(text_filepath, 'rb') as file:
            self.loaded_dict = pickle.load(file)
        self.ids = list(self.loaded_dict.keys())
         # can change this to pass more/fewer sequences
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
       
 
    def __len__(self):
        return len(self.ids)
   

    def __getitem__(self, idx):
        
        sequence = self.loaded_dict[self.ids[idx]]['sequence']
        text = self.loaded_dict[self.ids[idx]]['function']
        text_input = self.text_tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")  
        #print(f"Shape of function tokens = {function_tokens['input_ids'].shape}")
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding='max_length', truncation=True, return_tensors="pt")    
        sequence_input['input_ids'] = torch.squeeze(sequence_input['input_ids'])
        sequence_input['attention_mask'] = torch.squeeze(sequence_input['attention_mask'])
        text_input['input_ids'] = torch.squeeze(text_input['input_ids'])
        text_input['token_type_ids'] = torch.squeeze(text_input['token_type_ids'])
        text_input['attention_mask'] = torch.squeeze(text_input['attention_mask'])
        
        return sequence_input, text_input


class StructureDataset(Dataset):

    def __init__(self, folder_path="/p/scratch/hai_oneprot/alphafold_swissprot/zipped", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
         
        with open('/p/scratch/hai_oneprot/structure_sequence.pkl', 'rb') as file:
            self.loaded_data = pickle.load(file)
        self.id_list = list(self.loaded_data.keys())
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)
    def __len__(self):
        return len(self.id_list)
   

    def __getitem__(self, idx):
        
        atom_pos, atom_names, atom_amino_id, amino_types, aminoTypeSingleLetter_  = etract_chain_from_pdb(self.loaded_data[self.id_list[idx]]['file_path'])
        sequence = self.loaded_data[self.id_list[idx]]['sequence']
       
        structure_input = protein_to_graph(np.squeeze(amino_types), np.squeeze(atom_amino_id), np.squeeze(atom_names), np.squeeze(atom_pos))
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding='max_length', truncation=True, return_tensors="pt")    
        sequence_input['input_ids'] = torch.squeeze(sequence_input['input_ids'])
        sequence_input['attention_mask'] = torch.squeeze(sequence_input['attention_mask'])
        return sequence_input, structure_input


class GODataset(Dataset):
    def __init__(self, go_filepath="/p/scratch/hai_oneprot/go_data_dict.pkl", sequence_tokenizer="facebook/esm2_t33_650M_UR50D"):
                
        with open(go_filepath, 'rb') as file:
            self.loaded_dict = pickle.load(file)
        
        go_embs = np.load('/p/project/hai_oneprot/merdivan1/embeddings.npz', allow_pickle=True)['embds'].item()
 
        self.go_emb_token = {}
        ind = 2
        for go_emb_key, _ in go_embs.items():
    
            self.go_emb_token[go_emb_key] = ind
            ind = ind +1
            
        self.ids = list(self.loaded_dict.keys())
         # can change this to pass more/fewer sequences
        self.sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_tokenizer)

    def __len__(self):
        return len(self.ids)
        

    def __getitem__(self, idx):
        
        sequence = self.loaded_dict[self.ids[idx]]['sequence']
        go_terms = self.loaded_dict[self.ids[idx]]['GO']
        go_token_list = go_terms.strip().split('; ')

        
        go_embs_pad = torch.zeros((511, 200))
        goterm_mask = torch.zeros((512))
        goterm_mask[0] = 1 # cls token mask should be 1
        counter = 0
        for go_term in go_token_list:
            if go_term in self.go_embs and counter<512:
                go_embs_pad[counter:counter+1,:] = torch.tensor(np.array(self.go_embs[go_term]))
                goterm_mask[counter+1:counter+2] = 1
                counter = counter + 1
        
        go_input = {'inputs_embeds':go_embs_pad, 'attention_mask':goterm_mask}
        sequence_input = self.sequence_tokenizer(sequence, max_length=1026, padding='max_length', truncation=True, return_tensors="pt")    
        sequence_input['input_ids'] = torch.squeeze(sequence_input['input_ids'])
        sequence_input['attention_mask'] = torch.squeeze(sequence_input['attention_mask'])
        return sequence_input, go_input


def structure_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    sequences, structures = zip(*data)
    
    sequence_inputs = []
    sequence_atts = []
    for sequence in sequences:
        sequence_inputs.append(torch.unsqueeze(sequence['input_ids'], dim=0)) 
        sequence_atts.append(torch.unsqueeze(sequence['attention_mask'], dim=0))
    
    sequence_input = {}
    sequence_input['input_ids'] = torch.cat(sequence_inputs, dim=0)
    sequence_input['attention_mask'] = torch.cat(sequence_atts, dim=0)

    return sequence_input, Batch.from_data_list(structures)



'''

def text_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_model)
    sequences, functions = zip(*data)
    #print(f"Length of Sequences = {len(sequences)}")
    #print(f"Length of Functions = {len(functions)}")
    
    
    text_tokens = text_tokenizer(functions, max_length=512, padding=True, truncation=True, return_tensors="pt")  
    #print(f"Shape of function tokens = {function_tokens['input_ids'].shape}")
    sequence_tokens = sequence_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt")    
    #print(f"Shape of sequence tokens = {sequence_tokens['input_ids'].shape}")

    return sequence_tokens, text_tokens



def go_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    sequences, go_terms = zip(*data)
    #print(f"Length of Sequences = {len(sequences)}")
    #print(f"Length of GOs = {len(go_terms)}")
    sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_model)
    sequence_tokens = sequence_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt")    
    #print(f"Shape of sequence tokens = {sequence_tokens['input_ids'].shape}")
    
    max_length = 0
    for go_term in go_terms:
        if len(go_term)>max_length:
            max_length=len(go_term)
    
    #print(f"Max Go Length {max_length}")
    go_embs_pad = torch.zeros((len(go_terms), max_length, 200))
    goterm_mask = torch.zeros((len(go_terms), max_length+1))
    for ind, go_term in enumerate(go_terms):
        go_embs_pad[ind,:len(go_term),:] = torch.tensor(np.array(go_term))
        goterm_mask[ind,:len(go_term)+1] = 1
    #print(f"Shape Go Emb {go_embs_pad.shape}")
    go_embs = {'inputs_embeds':go_embs_pad, 'attention_mask':goterm_mask}

    return sequence_tokens, go_embs


def msa_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1024)
    sequence_tokenizer = AutoTokenizer.from_pretrained(sequence_model)
    sequences, msa_datas = zip(*data)
    #print(f"Length of Sequences = {len(sequences)}")
    #print(f"Length of MSAs = {len(msa_datas)}")
    
    msa_token_list = []
    max_len = 0
    max_depth = 0
    
    _, _, msa_tokens = msa_transformer_batch_converter(msa_datas)
    if msa_tokens.shape[2]>1024:
        msa_tokens = msa_tokens[:,:,:1024]

    sequence_tokens = sequence_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt")    
    #print(f"Shape of sequence tokens = {sequence_tokens['input_ids'].shape}")

    return sequence_tokens, msa_tokens
'''
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
from unicore.data import LMDBDataset
from unicore.data.dictionary import Dictionary
import os
from unimol.tasks.unimol_pocket import UniMolPocketTask



class MSADataset(Dataset):
    def __init__(self, data_dir="/p/scratch/hai_oneprot/Dataset_25_06_24/", split='train', max_length=1024, msa_depth=100, seq_tokenizer="facebook/esm2_t33_650M_UR50D",seqsim='30ss'):
        filename=f"{data_dir}/{seqsim}/{split}_msa.csv"
        self.msa_files = filter_and_create_msa_file_list(filename)
        _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_padding_idx =1
        self.msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1022)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
        self.max_length = max_length
        self.msa_depth = msa_depth
        self.split=split
 
    def __len__(self):
        return len(self.msa_files)
       
    def __getitem__(self, idx):
        return self.msa_files[idx]


    def collate_fn(self, msa_files):
        """
        data: is a list of tuples with (example, label, length)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        sequences = []
        msas = []
        for msa_file in msa_files:
            msa_data = read_msa(msa_file)
            msa_data = greedy_select(msa_data, num_seqs=self.msa_depth)
            sequence = msa_data[0][1]
            sequences.append(sequence)
            msas.append(msa_data)

        _, _, msa_input = self.msa_transformer_batch_converter(msas)
        msa_input=msa_input[:,:,:self.max_length]
        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids   
        
        return sequence_input.long(), torch.as_tensor(msa_input).long()

class StructDataset(Dataset):

    def __init__(self, data_dir = '/p/scratch/hai_oneprot/Dataset_25_06_24', split="train", seq_tokenizer="facebook/esm2_t33_650M_UR50D", use_struct_mask=False, use_struct_coord_noise=False, use_struct_deform=False,pockets=False,seqsim='30ss'):
        self.id_list = []
        self.pockets=pockets
        self.split = split
        if not pockets:
            self.h5_file =  f'{data_dir}/merged.h5'
        else:
            self.h5_file = f'{data_dir}/pockets_100_residues.h5'

        self.use_struct_mask = use_struct_mask
        self.use_struct_coord_noise = use_struct_coord_noise
        self.use_struct_deform = use_struct_deform

        if self.pockets:
            with open(f'{data_dir}/{seqsim}/{split}_pocket.csv', 'r') as file:

                for line in file:
                    self.id_list.append(line.split(',')[0].strip()) 
        else:
            with open(f'{data_dir}/{seqsim}/{split}_seqstruc.csv', 'r') as file:

                for line in file:
                    self.id_list.append(line.split(',')[0].strip())  

        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)

    def __len__(self):
        return len(self.id_list)
        
    def __getitem__(self, idx):       
        return self.id_list[idx]

    def collate_fn(self, seq_ids, return_raw_sequences=False):
        """
        data: is a list of tuples with (example, label, length)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        sequences, structures = [], []
        for seq_id in seq_ids:
            with h5py.File(self.h5_file, 'r') as file:
                sequence = file[seq_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')
                sequences.append(sequence)
            structures.append(protein_to_graph(seq_id, self.h5_file, 'non_pdb' , 'A', pockets=self.pockets))

        if return_raw_sequences:
            return sequences
        
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
    

def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def collate_tokens_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if values[0].shape[1]<values[0].shape[0]:
        size2=values[0].shape[1]
    else:
        size2=size

    size = size if pad_to_length is None else max(size, pad_to_length)

    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        
    res = values[0].new(len(values), size, size2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:v.shape[0], :v.shape[1]])
    return res




class PocketDataset(Dataset):
    
    def __init__(self, data_dir='/p/scratch/found/structures/swissprot/',
                 split='train',train_name='_pockets',filename='AlphaFold_swiss_v4.h5',
                 seq_tokenizer="facebook/esm2_t33_650M_UR50D",data_type='h5',
                 train_subset='train',valid_subset='val',
                 mask_prob=0.15, leave_unmasked_prob=0.05,
                 random_token_prob=0.05, noise_type='uniform',
                 noise=1.0,remove_hydrogen=False,remove_polar_hydrogen=False,
                 max_atoms=256, dict_name='dict_coarse.txt',
                 max_seq_len=512):
        
        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
    
        args=Namespace( 
         seed=1,  
         train_subset=train_subset, 
         valid_subset=valid_subset, 
         data=data_dir, 
         mask_prob=mask_prob, 
         leave_unmasked_prob=leave_unmasked_prob, 
         random_token_prob=random_token_prob, 
         noise_type=noise_type, 
         noise=noise, 
         remove_hydrogen=remove_hydrogen, 
         remove_polar_hydrogen=remove_polar_hydrogen, 
         max_atoms=max_atoms, 
         dict_name=dict_name, 
         max_seq_len=max_seq_len,
         train_name=train_name
         )

        dictionary = Dictionary.load(os.path.join(data_dir,dict_name))
        self.data_type=data_type
        self.split=split
        if data_type=='lmdb':
            if split=="train":
                subset=train_subset.split(",")[0]
            elif split=="val":
                subset=valid_subset.split(",")[0]
            else:
                subset="test"
        else:
            if split=="train":
                subset=train_subset.split(",")[0]
            else:
                subset=split+'_pockets'

        
        task=UniMolPocketTask(args, dictionary)
        task.load_dataset(subset, combine=False, epoch=1,data_type=data_type)
        self.dataset = task.dataset(subset)
        # if data_type=='h5':
        self.h5_file = f'{data_dir}{filename}'
        if split=="train":
            meta_file = f'{data_dir}{split}{train_name}.csv'
        else:
            meta_file = f'{data_dir}{split}.csv'
    
        self.meta_data = list(pd.read_csv(meta_file)['name'])
        
        with h5py.File("/p/project/hai_oneprot/bazarova1/pockets3.h5",'r') as file:
            self.keys=list(file.keys())
        
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)


    def __len__(self):
        if self.split=='train':
            return len(self.dataset)
        else:
            return 5000
        #return len(self.dataset)

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, data):
        sequences, pockets = [], []

        if isinstance(data[0], str):
            uniprot_idx_list = [m.split("-")[1] for m in self.meta_data]
            data = [uniprot_idx_list.index(d) for d in data]

        for i in data:
            data1=self.dataset[i]
            if self.data_type=='h5':
                with h5py.File(self.h5_file, 'r') as file:
                    for chain in file[f'{self.meta_data[i]}']['structure']['0'].keys():
                        sequence = file[f'{self.meta_data[i]}']['structure']['0'][f'{chain}']['residues']['seq1'][()]
                        sequences.append(str(sequence))
                
                pocket=dict()
                pocket['src_tokens']=data1['net_input.src_tokens']
                pocket['src_distance']=data1['net_input.src_distance']
                pocket['src_coord']=data1['net_input.src_coord']
                pocket['src_edge_type']=data1['net_input.src_edge_type']
                pockets.append(pocket)
            else:
                with h5py.File("/p/project/hai_oneprot/bazarova1/pockets3.h5",'r') as file:
                    sequence=file[data1['target.pdb_id']][()]
                    sequences.append(str(sequence))
                    pocket=dict()
                    pocket['src_tokens']=data1['net_input.src_tokens']
                    pocket['src_distance']=data1['net_input.src_distance']
                    pocket['src_coord']=data1['net_input.src_coord']
                    pocket['src_edge_type']=data1['net_input.src_edge_type']
                    pockets.append(pocket)
                
            pocket_input={key: collate_tokens_2d([d[key] for d in pockets],0) for key in ['src_distance','src_coord','src_edge_type']}
            pocket_input['src_tokens']=collate_tokens([d['src_tokens'] for d in pockets],0)

        sequence_input = self.seq_tokenizer(sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt").input_ids
        
        return sequence_input.long(), pocket_input


#import random
class TextDataset(Dataset):
    def __init__(self, data_dir = '/p/scratch/hai_oneprot/Dataset_25_06_24' , split="train", text_tokenizer="allenai/scibert_scivocab_uncased", #"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                  seq_tokenizer="facebook/esm2_t33_650M_UR50D",seqsim='30ss'):
        self.h5_file=f'{data_dir}/merged.h5'
        self.split=split
        self.df=pd.read_csv(f'/p/scratch/hai_oneprot/Dataset_25_06_24/{seqsim}/{split}_text.csv',header=None)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.seq_tokenizer = AutoTokenizer.from_pretrained(seq_tokenizer)
       
 
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        prot_id = self.df[0].iloc[idx]

        with h5py.File(self.h5_file, 'r') as file:
            sequence = file[prot_id]['structure']['0']['A']['residues']['seq1'][()].decode('utf-8')

        text = self.df[1].iloc[seq_ids]
        return text, sequence

    def collate_fn(self, data):
        texts = [d[0] for d in data]
        sequences = [d[1] for d in data]

        text_input = self.text_tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids   
        sequence_input = self.seq_tokenizer(sequences, max_length=1026, padding=True, truncation=True, return_tensors="pt").input_ids   
        return sequence_input.long(), text_input.long()
 

'''
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

from src.data.components.utils import greedy_select, extract_text_between_words, read_msa
from tqdm import tqdm
import os
import pickle
import esm
import h5py
import torch

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
def create_file_list():
    file_list = []
    print("Creating File List")
    with open('/p/project/hai_oneprot/merdivan1/files_list.txt', 'r') as file:
        for line in file:
            filename = line.strip().split()[-1]  # Remove newline characters
            if (".a3m" in filename and "/a3m/" in filename):
                file_list.append(os.path.join("/p/project/hai_oneprot/openfold", filename))



    print("Filtered filenames:")
    print(file_list[:10])
    print(file_list[-10:])
    print("Total File Number")
    print(len(file_list))
    return file_list[0:50000]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter(truncation_seq_length=1024)
msa_transformer.eval() 
msa_transformer.to(device)


file_list = create_file_list()


# Path to your HDF5 file
h5_file_path = '/p/scratch/hai_oneprot/precomputed/sequence_msa_0_50k.h5'

counter = 0
existing_counter = 0
with h5py.File(h5_file_path, 'w') as h5_file:
    
    for filename in tqdm(file_list, miniters=1000):

        _id = extract_text_between_words(filename, "/p/project/hai_oneprot/", "/a3m/")
        _id = _id.replace("/","_")

        
        if _id in h5_file:
            #print(f"ID {_id} already exists, skipping.")
            existing_counter = existing_counter+1
            continue
        else:
            msa_data = read_msa(filename)
            msa_data = greedy_select(msa_data, num_seqs=256)
        
            id_group = h5_file.create_group(_id)
            id_group.attrs['sequence'] = msa_data[0][1]
            
            _, _, msa_tokens = msa_transformer_batch_converter([msa_data])

            msa_tokens = msa_tokens.to(device)
            with torch.no_grad():
                modality_output = msa_transformer(msa_tokens[:,:,:1024],repr_layers=[12])

            msa_output = modality_output['representations'][12][:,0,:,:]

            # Store the MSA as a torch tensor
            id_group.create_dataset('msa', data=msa_output.detach().cpu().numpy())
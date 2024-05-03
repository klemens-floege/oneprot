'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyMolIO.py

    \brief Functions to load protein files.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as np
from collections import defaultdict
from Bio import SeqIO
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import string
from scipy.spatial.distance import  cdist
import os
import pickle
import torch


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter 
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def extract_text_between_words(input_string, start_word, end_word):
    start_index = input_string.find(start_word)
    end_index = input_string.find(end_word, start_index + len(start_word))
    
    if start_index != -1 and end_index != -1:
        extracted_text = input_string[start_index + len(start_word):end_index].strip()
        return extracted_text
    else:
        return None
    

def filter_and_create_msa_file_list(file_path):
    file_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if ".a3m" in line:
                file_list.append(line)
#                file_list.append(line.split(',')[1])             
    
    return file_list

def get_sequence_and_msa(filename):

        msa_data = read_msa(filename)
        return {'sequence':msa_data[0][1], 'MSA': msa_data }

def process_files_and_save_dict(file_list):
    data = {}
    ids = []
    counter = 0
    
    for filename in file_list:
        try:
            if counter % 10000 == 0:
                print(counter)
            
            name = extract_text_between_words(filename, "/p/project/hai_oneprot/", "/a3m/")
            name = name.replace("/", "_")
            ids.append(name)
            msa_data = read_msa(filename)  # Define read_msa function
            
            data[name] = {'sequence': msa_data[0][1], 'MSA': msa_data}
            counter = counter + 1
        except:
            print(f"File issue {filename}")
    
    # Save dictionary to msa_data_dict.pkl file
    with open('/p/scratch/hai_oneprot/msa_data_dict.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('Dictionary saved successfully to file')

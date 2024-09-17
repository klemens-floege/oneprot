import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist
import string
from Bio import SeqIO
import os


def filter_and_create_msa_file_list(filename: str) -> List[str]:
    file_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if ".a3m" in line:
                line=line.split(',')[1]
                file_list.append(line)
    
    return file_list

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

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

def read_msa(msa_file: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    try:
        temp= [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]
    except:
        temp= [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename+'.a3m', "fasta")]
    return temp
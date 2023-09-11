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
from src.data.components.PyPeriodicTable import PyPeriodicTable
import os
import pickle
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

from torch import nn


def load_protein_pdb(pDBFilePath, pLoadAnim = True, pLoadHydrogens = False, 
    pLoadH2O = False, pLoadGroups = True, pChainFilter = None):
    """Method to load a protein from a PDB file.

    Args:
        pDBFilePath (string): File path to the pdb.
        pLoadAnim (bool): Boolean that indicates if we load the animation.
    """

    atomPos = []
    atomTypes = []
    atomNames = []
    atomResidueIndexs = [] 
    atomResidueType = [] 
    atomChainName = []

    # Parse the PDB file.
    with open(pDBFilePath, 'r') as pdbFile:
        auxAtomResidueIds = []
        auxAltLoc = ""
        dictInsCodes = {}
        for line in pdbFile:

            # Start of a new model.
            if line.startswith("MODEL"):
                atomPos.append([])

            if line.startswith("ENDMDL"):
                # If we only need the first frame.
                if not(pLoadAnim):
                    break
                # If the number of atoms does not match between key frames.
                elif len(atomPos) > 1 and len(atomPos[-1]) != len(atomPos[-2]):
                    atomPos = atomPos[:-1]
                    break

            # Process an atom
            if line.startswith("ATOM") or (line.startswith("HETATM") and pLoadGroups):
                curChainName = line[21:22].strip()
                if pChainFilter is None or curChainName == pChainFilter:

                    curAtomLabel = line[76:78].strip()

                    # If the atom type is not provided, use the first letter of the atom name.
                    # This could lead to errors but the PDB should have the atom type in the 
                    # first place.
                    if len(curAtomLabel) == 0:
                        curAtomLabel = line[12:16].strip()[0]

                    if curAtomLabel != 'H' or pLoadHydrogens:
                        curResidueLabel = line[17:20].strip()

                        if curResidueLabel != 'HOH' or pLoadH2O:
                            if len(atomPos) == 0:
                                atomPos.append([])
                            
                            #Check that is not alternate location definition.
                            curAltLoc = line[16:17]
                            validPosAltLoc = (curAltLoc == " ")
                            if not validPosAltLoc:
                                if auxAltLoc == "":
                                    auxAltLoc = curAltLoc
                                    validPosAltLoc = True
                                else:
                                    validPosAltLoc = (curAltLoc == auxAltLoc)

                            #Check for the  insertion code of the Residue.
                            curICode = line[26:27]
                            curResidueIndex = int(line[22:26].strip())
                            if curResidueIndex in dictInsCodes:
                                validPosICode = dictInsCodes[curResidueIndex] == curICode
                            else:
                                dictInsCodes[curResidueIndex] = curICode
                                validPosICode = True

                            if validPosAltLoc and validPosICode:
                                # Save the atom position.
                                atomXCoord = float(line[30:38].strip())
                                atomYCoord = float(line[38:46].strip())
                                atomZCoord = float(line[46:54].strip())
                                atomPos[-1].append([atomXCoord, atomYCoord, atomZCoord])

                                # Save the topology only in the first model.
                                if len(atomPos) == 1:
                                    # Save the Residue types
                                    atomResidueType.append(curResidueLabel)

                                    # Save the Residue index.
                                    atomResidueIndexs.append(curResidueIndex)

                                    # Save the atom type.
                                    atomTypes.append(curAtomLabel)
                                    
                                    # Save the atom name.
                                    atomNames.append(line[12:16].strip())

                                    # Save the chain name.
                                    if len(curChainName) == 0:
                                        curChainName = 'Z'
                                    atomChainName.append(curChainName)

        # If no atom positions are loaded raise exception.
        if len(atomPos) == 0:
            raise Exception('Empty pdb')
        if len(atomPos[0]) == 0:
            raise Exception('Empty pdb')

        # Transform to numpy arrays
        atomPos = np.array(atomPos)
        atomTypes = np.array(atomTypes)
        atomNames = np.array(atomNames)
        atomResidueIndexs = np.array(atomResidueIndexs)
        atomResidueType = np.array(atomResidueType)
        atomChainName = np.array(atomChainName)

        # Center the molecule
        coordMax = np.amax(atomPos[0], axis=(0))
        coordMin = np.amin(atomPos[0], axis=(0))
        center = (coordMax + coordMin)*0.5
        atomPos = atomPos - center.reshape((1, 1, 3))

    return atomPos, atomTypes, atomNames, atomResidueIndexs, atomResidueType, atomChainName, center

def save_protein_pdb(pFilePath, pProtein):
    """Method to save a protein to a PDB file.

    Args:
        pFilePath (string): Path to the file.
        pProtein (MCPyProtein): Protein to save.
    """
    with open(pFilePath, 'w') as protFile:
        aminoCounter = 0
        lastChainName = "A"
        lastAminoId = -1
        for curAtomIter in range(len(pProtein.atomTypes_)):
            curAtomName = pProtein.atomNames_[curAtomIter]
            while len(curAtomName) < 3:
                curAtomName = curAtomName+" "
            aminoIds = pProtein.atomAminoIds_[curAtomIter]
            resName = pProtein.atomResidueNames_[curAtomIter]
            chainName = pProtein.atomChainNames_[curAtomIter]
            if lastAminoId != aminoIds:
                if lastChainName != chainName:
                    aminoCounter +=2
                else:
                    aminoCounter +=1
            lastChainName = chainName
            lastAminoId = aminoIds
            curAtomPosition = pProtein.atomPos_[0, curAtomIter] + pProtein.center_
            xPosText = "{:8.3f}".format(curAtomPosition[0])
            yPosText = "{:8.3f}".format(curAtomPosition[1])
            zPosText = "{:8.3f}".format(curAtomPosition[2])
            occupancy = "  1.00"
            tempFactor = "  1.00"
            atomType = pProtein.periodicTable_.labels_[pProtein.atomTypes_[curAtomIter]].split("/")[0]
            protFile.write(("ATOM  %5d  %s %s %s%4d    %s%s%s%s%s           %s\n")%(curAtomIter+1, 
                curAtomName, resName, chainName, aminoCounter,
                xPosText, yPosText, zPosText, occupancy, tempFactor, atomType))


def save_molecule_pdb(pFilePath, pMolecule):
    """Method to save a molecule to a PDB file.

    Args:
        pFilePath (string): Path to the file.
        pMolecule (MCPyMolecule): Molecule to save.
    """
    with open(pFilePath, 'w') as protFile:
        for curAtomIter in range(len(pMolecule.atomTypes_)):
            if not pMolecule.atomNames_ is None:
                curAtomName = pMolecule.atomNames_[curAtomIter]
                while len(curAtomName) < 3:
                    curAtomName = curAtomName+" "
            else:
                curAtomName = "  X"
            
            aminoType = "XXX"
            chainName = "A"
            
            curAtomPosition = pMolecule.atomPos_[0, curAtomIter] + pMolecule.center_
            xPosText = "{:8.3f}".format(curAtomPosition[0])
            yPosText = "{:8.3f}".format(curAtomPosition[1])
            zPosText = "{:8.3f}".format(curAtomPosition[2])
            occupancy = "  1.00"
            tempFactor = "  1.00"
            aminoCounter = 1
            atomType = pMolecule.periodicTable_.labels_[pMolecule.atomTypes_[curAtomIter]].split("/")[0]
            protFile.write(("ATOM  %5d  %s %s %s%4d    %s%s%s%s%s           %s\n")%(curAtomIter+1, 
                curAtomName, aminoType, chainName, aminoCounter,
                xPosText, yPosText, zPosText, occupancy, tempFactor, atomType))


def load_protein_mol2(pFilePath, pLoadHydrogens = False, pLoadH2O = False, 
    pLoadGroups = True, pChainFilter = None):
    """Method to load a protein from a Mol2 file.

    Args:
        pFilePath (string): File path to the pdb.
    """

    with open(pFilePath, 'r') as datasetFile:
        # Read the lines of the file.
        lines = datasetFile.readlines()

        # Get the overall information of the molecule.
        splitInitLine = lines[2].split()
        
        # Iterate over the lines.
        atomPos = []
        atomTypes = []
        atomNames = []
        atomResidueIndexs = [] 
        atomResidueName = []
        residueDict = {}
        residueIndexDict = {}
        atomSection = False
        structureSection = False
        for curLine in lines:

            # Check if it is the start of a new section.
            if curLine.startswith("@<TRIPOS>ATOM"):
                atomSection = True
                structureSection = False
            elif curLine.startswith("@<TRIPOS>SUBSTRUCTURE"):
                atomSection = False
                structureSection = True
            elif curLine.startswith("@<TRIPOS>"):
                atomSection = False
                structureSection = False
            else:

                # If we are in the atom section.
                if atomSection:
                    lineElements = curLine.rstrip().split()
                    curAtomName = lineElements[1]
                    curAtomPos = [float(lineElements[2]), 
                        float(lineElements[3]), 
                        float(lineElements[4])]
                    curAtomType = lineElements[5].split('.')[0].upper()
                    if curAtomType != 'H' or pLoadHydrogens:
                        curResidueName = lineElements[7][0:3].upper()
                        if curResidueName != 'HOH' or pLoadH2O:
                            curResidueIndex = int(lineElements[6])

                            # Check if the Residue is valid (it did not appear before).
                            if not(curResidueIndex in residueIndexDict):
                                curVector = []
                                residueIndexDict[curResidueIndex] = []
                            else:
                                curVector = residueIndexDict[curResidueIndex]

                            if not(curAtomName in curVector) and (len(curVector) == 0 or curVector[-1] != '-1'):
                                if curAtomType != "DU":

                                    # Update the temporal dictionary.
                                    residueIndexDict[curResidueIndex].append(curAtomName)
                                    
                                    # Store the atom.
                                    atomPos.append(curAtomPos)
                                    atomTypes.append(curAtomType)
                                    atomNames.append(curAtomName)
                                    atomResidueIndexs.append(curResidueIndex)
                                    atomResidueName.append(curResidueName)
                            else:
                                # Update the temporal dictionary.
                                residueIndexDict[curResidueIndex].append('-1')

                # If we are in the structure section.
                elif structureSection:
                    lineElements = curLine.rstrip().split()
                    if lineElements[3] == "RESIDUE" or pLoadGroups:
                        residueDict[lineElements[0]] = (lineElements[5], lineElements[6])

        # Prepare the final arrays.
        atomResidueType = [] 
        atomChainName = []
        auxAtomMask = np.full((len(atomTypes)), False, dtype=bool)
        for curAtomIter, curResidueIndex in enumerate(atomResidueIndexs):
            curKey = str(curResidueIndex)
            if curKey in residueDict:
                curResidue = residueDict[curKey]
                if pChainFilter is None or pChainFilter == curResidue[0]:
                    atomResidueType.append(curResidue[1])
                    atomChainName.append(curResidue[0])
                    auxAtomMask[curAtomIter] = True


        # Transform to numpy arrays
        atomPos = np.array(atomPos)
        atomTypes = np.array(atomTypes)
        atomNames = np.array(atomNames)
        atomResidueIndexs = np.array(atomResidueIndexs)
        atomResidueType = np.array(atomResidueType)
        atomChainName = np.array(atomChainName)

        atomPos = atomPos[auxAtomMask]
        atomTypes = atomTypes[auxAtomMask]
        atomNames = atomNames[auxAtomMask]
        atomResidueIndexs = atomResidueIndexs[auxAtomMask]

        # Center the molecule
        coordMax = np.amax(atomPos, axis=(0))
        coordMin = np.amin(atomPos, axis=(0))
        center = (coordMax + coordMin)*0.5
        atomPos = atomPos - center.reshape((1, 3))
        atomPos = atomPos.reshape((1, -1, 3))

    return atomPos, atomTypes, atomNames, atomResidueIndexs, atomResidueType, atomChainName, center

def extract_chain_from_pdb(pdb_path, chain=None):

    atomPos, atomTypes, atomNames, atomResidueIds, atomResidueType, \
                    atomChainName, transCenter = load_protein_pdb(pdb_path, pChainFilter = chain)
    periodicTable_ = PyPeriodicTable()
    auxAtomTypes = np.array([periodicTable_.get_atom_index(curIndex) \
            for curIndex in atomTypes])
    maskValidAtoms = auxAtomTypes >= 0
    auxAtomTypes = auxAtomTypes[maskValidAtoms]


    atomPos_ = atomPos[:, maskValidAtoms]
    atomNames_ = atomNames[maskValidAtoms]
    atomTypes_ = auxAtomTypes
    atomChainNames_ = atomChainName[maskValidAtoms]
    chainNames = np.unique(atomChainNames_)
    atomChainIds_ = np.array([np.where(chainNames == curChainName)[0][0] \
        for curChainName in atomChainNames_])

    # Compute residue ids.
    auxResidueIds = atomResidueIds[maskValidAtoms] - np.amin(atomResidueIds[maskValidAtoms]) + 1
    auxResidueIds = auxResidueIds + atomChainIds_*np.amax(auxResidueIds)
    _, atomResidueIds_ = np.unique(auxResidueIds, return_inverse=True)
    auxResidueTypes = atomResidueType[maskValidAtoms]
    atomResidueNames_ = auxResidueTypes

    # Convert the aminoacid label to id.
    auxResidueTypes = np.array([periodicTable_.get_aminoacid_index(curIndex) \
        for curIndex in auxResidueTypes])

       
    auxResidueTypesSingleLetter = np.array([periodicTable_.get_aminoacid_letter(curIndex) \
        for curIndex in auxResidueTypes])
    
    # Get aminoacid information.
    mask = np.logical_and(atomNames_== "CA", atomTypes_ == 5)
    aminoPos_ = atomPos_[:, mask]
    aminoType_ = auxResidueTypes[mask]
    aminoTypeSingleLetter_ = auxResidueTypesSingleLetter[mask]
    aminoChainIds_ = atomChainIds_[mask].reshape((-1))
    # Process the amino ids.
    aminoOrigIds = auxResidueIds[mask]
    atomAminoIds_ = np.array([np.where(aminoOrigIds == curAminoId)[0] \
        for curAminoId in auxResidueIds])
    atomAminoIds_ = np.array([-1 if len(curIndex)==0 else curIndex[0]
        for curIndex in atomAminoIds_])
    #h5File = h5py.File("/p/scratch/hai_oneprot/h5/101MA.hdf5", "w")
    # Save atoms.
    auxAtomNames = np.array([curName.encode('utf8') for curName in atomNames_])
    
 
    
    return atomPos_, auxAtomNames, atomAminoIds_, aminoType_, aminoTypeSingleLetter_
    
    '''
    h5File.create_dataset("atom_pos", data=atomPos_)
    h5File.create_dataset("atom_names", data=auxAtomNames)

    # Save additional atom info.

    h5File.create_dataset("atom_amino_id", data=atomAminoIds_)



    # Save aminoacids.

    h5File.create_dataset("amino_types", data=aminoType_)


    h5File.close()
    '''
    
def download_pdb(pdb_id: str,
                 download_dir: str,) -> None:
    r = get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
    open(os.path.join(download_dir, f"{pdb_id.upper()}.pdb"), 'w').write(r.text)
    return

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
    

def filter_and_create_msa_file_list(file_path='/p/project/hai_oneprot/merdivan1/files_list.txt'):
    file_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            filename = line.strip().split()[-1]  # Remove newline characters
            if ".a3m" in filename:
                file_list.append(os.path.join("/p/project/hai_oneprot/openfold", filename))
    
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

        
def _normalize(tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


def get_side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
def get_bb_embs(X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
def compute_diherals(v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
def protein_to_graph(amino_types, atom_amino_id, atom_names, atom_pos):
        #h5File = h5py.File(pFilePath, "r")
        data = Data()

        #amino_types = h5File['amino_types'][()] # size: (n_amino,)
        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        #atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        #atom_names = h5File['atom_names'][()] # size: (n_atom,)
        #atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        
        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = get_side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        # three backbone torsion angles
        bb_embs = get_bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = torch.unsqueeze(torch.tensor(amino_types),1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c

        assert len(data.x)==len(data.coords_ca)==len(data.coords_n)==len(data.coords_c)==len(data.side_chain_embs)==len(data.bb_embs)
        return data


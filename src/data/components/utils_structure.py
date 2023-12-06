
import h5py
import numpy as np
import logging

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from transformers import AutoTokenizer, EsmModel, EsmConfig

### Logging ###
'''
Initiate a logging method. Likely can remove instantiating and connect to full experiment log.
Or we can keep it seperate for handling the data
'''
logging.basicConfig(filename='create_dataset.log', encoding='utf-8', level=logging.DEBUG)
log = logging.getLogger(__name__) #for connecting to the logger

### One-hot encoding mapping for all natural amino acids

res2int = {b'ALA':0, b'ARG':1, b'ASN':2, b'ASP':3, b'CYS':4, b'GLN':5, b'GLU':6, b'GLY':7, b'HIS':8, b'ILE':9, b'LEU':10, b'LYS':11, b'MET':12, b'PHE':13, b'PRO':14, b'SER':15, b'THR':16, b'TRP':17, b'TYR':18, b'VAL':19, b'UNK':20}
res1int = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}

def get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names.astype('S'), b'N')
        mask_ca = np.char.equal(atom_names.astype('S'), b'CA')
        mask_c = np.char.equal(atom_names.astype('S'), b'C')
        mask_cb = np.char.equal(atom_names.astype('S'), b'CB')
        mask_g = np.char.equal(atom_names.astype('S'), b'CG') | np.char.equal(atom_names.astype('S'), b'SG') | np.char.equal(atom_names.astype('S'), b'OG') | np.char.equal(atom_names.astype('S'), b'CG1') | np.char.equal(atom_names.astype('S'), b'OG1')
        mask_d = np.char.equal(atom_names.astype('S'), b'CD') | np.char.equal(atom_names.astype('S'), b'SD') | np.char.equal(atom_names.astype('S'), b'CD1') | np.char.equal(atom_names.astype('S'), b'OD1') | np.char.equal(atom_names.astype('S'), b'ND1')
        mask_e = np.char.equal(atom_names.astype('S'), b'CE') | np.char.equal(atom_names.astype('S'), b'NE') | np.char.equal(atom_names.astype('S'), b'OE1')
        mask_z = np.char.equal(atom_names.astype('S'), b'CZ') | np.char.equal(atom_names.astype('S'), b'NZ')
        mask_h = np.char.equal(atom_names.astype('S'), b'NH1')

        _, atom_amino_id = np.unique(atom_amino_id, return_inverse=True)
       
        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the posit[f'{chain}']ion of C and N as the position of CA
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


def calc_side_chain_embs( pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        '''
        Calculate all side chain angles to add to the embeddings.
        '''
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

def _normalize(tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def calc_bb_embs(X):   
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


def protein_to_graph(identifier, h5_file='/p/scratch/hai_oneprot/alphafold_swiss_v4/AlphaFold_swiss_v4.h5', dataset='non_pdb', chain='A'):
        data = Data()
        #print(f"Entered with {identifier} {chain}")
        with h5py.File(h5_file, 'r') as file:

        #current implementation goes over all chains
        # amino_types = []
        # atom_amino_id = []
        # atom_names = []
        # atom_pos = []
        
                if chain == 'A' and dataset=='non_pdb':
                
                        amino_types = file[f'{identifier}']['structure']['0']['A']['residues']['seq1'][()]
                        amino_types = [res1int[AA] for AA in amino_types.decode('utf-8')] #size: (n_aa)

                        atom_amino_id = file[f'{identifier}']['structure']['0']['A']['polypeptide']['atom_amino_id'][()] #size: (n_atom,)
                        atom_names = file[f'{identifier}']['structure']['0']['A']['polypeptide']['type'][()] #size: (n_atom,)
                        atom_pos = file[f'{identifier}']['structure']['0']['A']['polypeptide']['xyz'][()] #size: (n_atoms, 3)
                elif chain == 'all' and dataset=='pdb':
                        
                        amino_types = []
                        atom_amino_id = []
                        atom_names = []
                        atom_pos = []

                        for chain_id in file[f'{identifier}']['structure']['0'].keys():
                                single_amino_types = file[f'{identifier}']['structure']['0'][f'{chain_id}']['residues']['seq1'][()]
                                single_amino_types = single_amino_types.decode('utf-8').replace('X','')
                                single_amino_types = [res1int[AA] for AA in single_amino_types] #size: (n_aa)
                                amino_types.extend(single_amino_types)

                                atom_amino_id.extend(file[f'{identifier}']['structure']['0'][f'{chain_id}']['polypeptide']['atom_amino_id'][()]) #size: (n_atom,)
                                atom_names.extend(file[f'{identifier}']['structure']['0'][f'{chain_id}']['polypeptide']['type'][()]) #size: (n_atom,)
                                atom_pos.extend(file[f'{identifier}']['structure']['0'][f'{chain_id}']['polypeptide']['xyz'][()]) #size: (n_atoms, 3)

                else:

                        amino_types = file[f'{identifier}']['structure']['0'][f'{chain}']['residues']['seq1'][()]
                        amino_types = amino_types.decode('utf-8').replace('X','')
                        amino_types = [res1int[AA] for AA in amino_types]

                        atom_amino_id = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['atom_amino_id'][()] #size: (n_atom,)
                        atom_names = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['type'][()] #size: (n_atom,)
                        atom_pos = file[f'{identifier}']['structure']['0'][f'{chain}']['polypeptide']['xyz'][()] #size: (n_atoms, 3)

        try:
                pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        except:
                print(f"Issue with {identifier} {chain} ")      
                   
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1


        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = calc_side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        # three backbone torsion angles
        bb_embs = calc_bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = torch.unsqueeze(torch.tensor(amino_types),1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c

        try:
            #print(identifier,len(data.x),len(data.coords_ca),len(data.coords_n),len(data.coords_c),len(data.side_chain_embs),len(data.bb_embs))
            assert len(data.x)==len(data.coords_ca)==len(data.coords_n)==len(data.coords_c)==len(data.side_chain_embs)==len(data.bb_embs)
        except AssertionError:
            print(identifier)
            raise AssertionError

        return data
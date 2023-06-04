import h5py
import torch as t
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torchmetrics.functional import pairwise_euclidean_distance
import time

atom_mapping = {0:'H', 1:'C', 2:'N', 3:'O', 4:'F', 5:'P', 6:'S', 7:'CL', 8:'BR', 9:'I', 10: 'UNK'}
atom_eye = t.eye(len(list(atom_mapping.keys())))

residue_mapping = {0:'ALA', 1:'ARG', 2:'ASN', 3:'ASP', 4:'CYS', 5:'CYX', 6:'GLN', 7:'GLU', 8:'GLY', 9:'HIE', 10:'ILE', 11:'LEU', 12:'LYS', 13:'MET', 14:'PHE', 15:'PRO', 16:'SER', 17:'THR', 18:'TRP', 19:'TYR', 20:'VAL', 21:'UNK'}

ligand_atoms_mapping = {8: 0, 16: 1, 6: 2, 7: 3, 1: 4, 15: 5, 17: 6, 9: 7, 53: 8, 35: 9, 5: 10, 33: 11, 26: 12, 14: 13, 34: 14, 44: 15, 12: 16, 23: 17, 77: 18, 27: 19, 52: 20, 30: 21, 4: 22, 45: 23}

class MdDataset(Dataset):
    def __init__(self, h5_path, cases,
    interaction_cutoff = 4.5, # distance in angstrom to define interacting atoms
    covalent_cutoff = 2.2
    ):
        super().__init__(h5_path)
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "r")
        self.keys = list(self.h5.keys())
        self.interaction_cutoff = interaction_cutoff
        self.covalent_cutoff = covalent_cutoff
        self.edge_types = {
            "covalent": 0,
            "same_chain": 1,
            "diff_chain": 2
        }
        kd_f = open("/data/kd.txt", "r")
        lines = [l.split() for l in kd_f]
        kd_raw = [(l[0], l[3].replace("Kd=", "")) for l in lines if "Kd=" in l[3]]
        kd = []
        ids = []
        for i,v in kd_raw:
            v = v.replace("M","").replace("n","e-9").replace("u","e-6")
            v = v.replace("m","e-3").replace("p","e-12").replace("Kd=","")
            v = v.replace("f", "e-15")
            kd.append(float(v))
            ids.append(i)
        # kd = t.tensor(kd)
        # kd = (kd - kd.min())/(kd.max() - kd.min())
        # self.max = kd.max()
        # self.min = kd.min()
        self.targets = dict(zip(ids, kd))
        self.cases = cases

    def len(self):
        return len(self.keys) 
    
    def get(self, idx):
        if self.cases[idx] not in list(self.targets.keys()):
            print(f"{idx} don't have a Kd value.")
        case = self.h5[self.cases[idx]]
        covalent_cutoff = self.covalent_cutoff
        interaction_cutoff = self.interaction_cutoff
        coordinates = t.as_tensor(case["trajectory_coordinates"][:])
        chain_index = t.as_tensor(case["molecules_begin_atom_index"][:])
        atoms = t.as_tensor(case["atoms_type"])
        self.atoms = atoms

        dist = t.stack([t.triu(pairwise_euclidean_distance(c)) for c in coordinates])
        dist = dist.transpose(0,1).mean(1) # dist is the mean dist between atoms for all MD frames
        self.dist = dist
        pos = coordinates[0]

        covalent_mask = (dist < covalent_cutoff) & (dist > 0)
        interaction_mask = (dist > covalent_cutoff) & (dist < interaction_cutoff)
        edge_mask = (interaction_mask | covalent_mask)

        edge_index = edge_mask.nonzero().t()
        edge_distance = dist[edge_mask]
        edge_covalent_mask = (edge_distance < covalent_cutoff)
        edge_samechain_mask = (
            (edge_index[0] <= chain_index[1]) & (edge_index[1] <= chain_index[1])
        ) | (
            ((edge_index[0] >= chain_index[1]) & (edge_index[0] <= chain_index[-1])) &
            ((edge_index[1] >= chain_index[1]) & (edge_index[1] <= chain_index[-1]))
        )
        edge_type = t.tensor([-1]).repeat(edge_index.shape[1])
        edge_type[edge_covalent_mask] = self.edge_types["covalent"]
        edge_type[~edge_samechain_mask] = self.edge_types["diff_chain"]
        edge_type[edge_samechain_mask] = self.edge_types["same_chain"]
        edge_attr = t.stack([edge_distance,edge_type], dim=-1)
        
        x = t.stack([self.atom_1hot(atom) for atom in atoms])
        return Data(
            x = x,
            edge_index = edge_index,
            edge_attr = edge_attr,
            pos = pos,
            target = self.targets[self.cases[idx]]
        )
    
    def atom_1hot(self, atom):
        atom = (10, atom)[atom in list(atom_mapping.keys())]
        self.atom = atom
        return atom_eye[atom]

if __name__ == "__main__":
    dataset = MdDataset("/data/MD.hdf5")
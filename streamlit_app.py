import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import xgboost as xgb
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool

# Khởi tạo featurizer
featurizer = MultiHotAtomFeaturizer.v2()
featurizer_bond = MultiHotBondFeaturizer()

# Hàm chuẩn hóa SMILES
def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()

    standardized_list = []
    for smi in tqdm(batch, desc='Processing . . .'):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                cleanup = rdMolStandardize.Cleanup(mol)
                normalized = rdMolStandardize.Normalize(cleanup)
                uncharged = uc.uncharge(normalized)
                fragment = uc.uncharge(rdMolStandardize.FragmentParent(uncharged))
                ionized = rdMolStandardize.Reionize(fragment)
                disconnected = md.Disconnect(ionized)
                tautomer = te.Canonicalize(disconnected)
                smiles = Chem.MolToSmiles(tautomer, isomericSmiles=False, canonical=True)
                standardized_list.append(smiles)
            else:
                standardized_list.append(None)
                st.write(f"Invalid SMILES: {smi}")
        except Exception as e:
            st.write(f"An error occurred with SMILES {smi}: {str(e)}")
            standardized_list.append(None)
    return standardized_list

# Hàm chuyển đổi SMILES thành đặc trưng (Morgan fingerprint)
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Hàm chuyển đổi SMILES thành đối tượng PyG
def smi_to_pyg(smi, y=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]

    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [featurizer(a) for a in mol.GetAtoms()]
    bond_features = [featurizer_bond(b) for b in bonds]

    data = Data(
        edge_index=torch.LongTensor(list(zip(*atom_pairs))),
        x=torch.FloatTensor(atom_features),
        edge_attr=torch.FloatTensor(bond_features),
        mol=mol,
        smiles=smi
    )

    if y is not None:
        data.y = torch.FloatTensor([[y]])

    return data

# Định nghĩa lớp MyDataset
class MyDataset(Dataset):
    def __init__(self, standardized):
        mols = [smi_to_pyg(smi, y=None) for smi in tqdm(standardized, total=len(standardized))]
        self.X = [m for m in mols if m]

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)

# Định nghĩa lớp MyConv
class MyConv(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_p, arch='GIN', mlp_layers=1):
        super().__init__()
        if arch == 'GIN':
            h = nn.Sequential()
            for _ in range(mlp_layers - 1):
                h.append(nn.Linear(node_dim, node_dim))
                h.append(nn.ReLU())
            h.append(nn.Linear(node_dim, node_dim))
            self.gine_conv = GINEConv(h, edge_dim=edge_dim)
            self.batch_norm = nn.BatchNorm1d(node_dim)
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.gine_conv(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# Định nghĩa lớp MyGNN
class MyGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_p, arch='GIN', num_layers=3, mlp_layers=1):
        super().__init__()
        self.convs = nn.ModuleList(
            [MyConv(node_dim, edge_dim, dropout_p=dropout_p, arch=arch, mlp_layers=mlp_layers

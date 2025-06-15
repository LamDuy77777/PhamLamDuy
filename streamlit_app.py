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
from torch_geometric.nn import GINEConv, global_mean_pool
import random

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load XGBoost model
@st.cache_resource
def load_xgb_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Define GIN model components
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

class MyGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_p, arch='GIN', num_layers=3, mlp_layers=1):
        super().__init__()
        self.convs = nn.ModuleList(
            [MyConv(node_dim, edge_dim, dropout_p=dropout_p, arch=arch, mlp_layers=mlp_layers)
             for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x

class MyFinalNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, arch, num_layers, dropout_mlp, dropout_gin, embedding_dim, mlp_layers, pooling_method):
        super().__init__()
        node_dim = (node_dim - 1) + 118 + 1
        edge_dim = (edge_dim - 1) + 21 + 1

        self.gnn = MyGNN(node_dim, edge_dim, dropout_p=dropout_gin, arch=arch, num_layers=num_layers, mlp_layers=mlp_layers)

        if pooling_method == 'mean':
            self.pooling_fn = global_mean_pool
        else:
            raise ValueError("Phương pháp pooling không hợp lệ")

        self.head = nn.Sequential(
            nn.BatchNorm1d(node_dim),
            nn.Dropout(p=dropout_mlp),
            nn.Linear(node_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_mlp),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x0 = F.one_hot(x[:, 0].to(torch.int64), num_classes=118+1).float()
        edge_attr0 = F.one_hot(edge_attr[:, 0].to(torch.int64), num_classes=21+1).float()
        x = torch.cat([x0, x[:, 1:]], dim=1)
        edge_attr = torch.cat([edge_attr0, edge_attr[:, 1:]], dim=1)

        node_out = self.gnn(x, edge_index, edge_attr)
        graph_out = self.pooling_fn(node_out, batch)
        return self.head(graph_out)

# Load GIN model
@st.cache_resource
def load_gin_model():
    node_dim = 72
    edge_dim = 14
    best_params = {
        'embedding_dim': 256,
        'num_layer': 7,
        'dropout_mlp': 0.28210247642451436,
        'dropout_gin': 0.12555795277599677,
        'mlp_layers': 2,
        'pooling_method': 'mean'
    }
    model = MyFinalNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        arch='GIN',
        num_layers=best_params['num_layer'],
        dropout_mlp=best_params['dropout_mlp'],
        dropout_gin=best_params['dropout_gin'],
        embedding_dim=best_params['embedding_dim'],
        mlp_layers=best_params['mlp_layers'],
        pooling_method=best_params['pooling_method']
    ).to(device)
    with open('GIN_597_562_cpu.pkl', 'rb') as f:
        state_dict = torch.load(f, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Standardize SMILES
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
        except Exception:
            standardized_list.append(None)
    return standardized_list

# Convert SMILES to Morgan fingerprint for XGBoost
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Convert SMILES to PyTorch Geometric data for GIN
featurizer = MultiHotAtomFeaturizer.v2()
featurizer_bond = MultiHotBondFeaturizer()

def smi_to_pyg(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [featurizer(a) for a in mol.GetAtoms()]
    bond_features = [featurizer_bond(b) for b in bonds]
    data = Data(
        edge_index=torch.tensor(list(zip(*atom_pairs)), dtype=torch.long).t().contiguous(),
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_attr=torch.tensor(bond_features, dtype=torch.float)
    )
    return data

# Define Dataset for GIN
class MyDataset(Dataset):
    def __init__(self, smiles_list):
        self.data_list = [smi_to_pyg(smi) for smi in smiles_list if smi_to_pyg(smi) is not None]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# Streamlit app
st.title("Dự đoán với Mô hình XGBoost và GIN")

# Input method
input_method = st.radio("Chọn cách nhập SMILES:", ("Nhập thủ công", "Tải lên file CSV"))

if input_method == "Nhập thủ công":
    smiles_input = st.text_area("Nhập SMILES (mỗi SMILES trên một dòng):")
    if st.button("Dự đoán") and smiles_input:
        smiles_list = smiles_input.split('\n')
else:
    uploaded_file = st.file_uploader("Tải lên file CSV chứa SMILES", type=["csv"])
    if uploaded_file and st.button("Dự đoán"):
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
        else:
            st.write("File CSV phải chứa cột 'SMILES'")
            smiles_list = None

# Common processing steps
if 'smiles_list' in locals() and smiles_list:
    with st.spinner("Đang chuẩn hóa SMILES..."):
        standardized_smiles = standardize_smiles(smiles_list)
    
    # Create a list of tuples (original_smiles, standardized_smiles)
    smiles_pairs = [(orig, std) for orig, std in zip(smiles_list, standardized_smiles)]
    
    # Filter valid SMILES
    valid_pairs = [(orig, std) for orig, std in smiles_pairs if std is not None]
    invalid_smiles = [orig for orig, std in smiles_pairs if std is None]
    
    if invalid_smiles:
        st.write("Các SMILES sau không hợp lệ và không thể xử lý:")
        for smi in invalid_smiles:
            st.write(smi)
    
    if valid_pairs:
        valid_orig, valid_std = zip(*valid_pairs)
        
        with st.spinner("Đang thực hiện dự đoán phân loại với XGBoost..."):
            features = [smiles_to_features(smi) for smi in valid_std]
            xgb_model = load_xgb_model()
            xgb_predictions = xgb_model.predict(np.array(features))
        
        # Filter SMILES classified as 1
        classified_as_1 = [(orig, std) for (orig, std), pred in zip(valid_pairs, xgb_predictions) if pred == 1]
        
        if classified_as_1:
            orig_as_1, std_as_1 = zip(*classified_as_1)
            
            with st.spinner("Đang chuyển đổi thành dữ liệu đồ thị cho GIN..."):
                dataset = MyDataset(std_as_1)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            with st.spinner("Đang thực hiện dự đoán pEC50 với GIN..."):
                gin_model = load_gin_model()
                gin_predictions = []
                for batch in dataloader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    gin_predictions.extend(pred.cpu().numpy().flatten())
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'SMILES gốc': orig_as_1,
                'SMILES chuẩn hóa': std_as_1,
                'Dự đoán XGBoost': [1] * len(orig_as_1),
                'Dự đoán pEC50 (GIN)': gin_predictions
            })
            st.write("Kết quả dự đoán (chỉ các chất được phân loại là 1):", result_df)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Tải xuống kết quả dưới dạng CSV",
                data=csv,
                file_name="du_doan.csv",
                mime="text/csv"
            )
        else:
            st.write("Không có SMILES nào được dự đoán là 1.")
    else:
        st.write("Không có SMILES hợp lệ để dự đoán.")

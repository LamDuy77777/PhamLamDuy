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
            [MyConv(node_dim, edge_dim, dropout_p=dropout_p, arch=arch, mlp_layers=mlp_layers)
             for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x

# Định nghĩa lớp MyFinalNetwork
class MyFinalNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, arch, num_layers, dropout_mlp, dropout_gin, embedding_dim, mlp_layers, pooling_method):
        super().__init__()
        node_dim = (node_dim - 1) + 118 + 1
        edge_dim = (edge_dim - 1) + 21 + 1

        self.gnn = MyGNN(node_dim, edge_dim, dropout_p=dropout_gin, arch=arch, num_layers=num_layers, mlp_layers=mlp_layers)

        if pooling_method == 'add':
            self.pooling_fn = global_add_pool
        elif pooling_method == 'mean':
            self.pooling_fn = global_mean_pool
        elif pooling_method == 'max':
            self.pooling_fn = global_max_pool
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

# Tải mô hình XGBoost
@st.cache_resource
def load_xgb_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Tải state_dict của mô hình GIN và khởi tạo mô hình
@st.cache_resource
def load_gin_model():
    with open('GIN_597_562 (1).pkl', 'rb') as f:
        state_dict = pickle.load(f)

    node_dim = 72  # Giá trị thực tế từ dữ liệu huấn luyện
    edge_dim = 14  # Giá trị thực tế từ dữ liệu huấn luyện
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
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Định nghĩa device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Giao diện ứng dụng
st.title("Dự đoán với Mô hình XGBoost và GIN")

# Chọn cách nhập SMILES
input_method = st.radio("Chọn cách nhập SMILES:", ("Nhập thủ công", "Tải lên file CSV"))

if input_method == "Nhập thủ công":
    smiles_input = st.text_area("Nhập SMILES (mỗi SMILES trên một dòng):")
    if st.button("Dự đoán") and smiles_input:
        smiles_list = smiles_input.split('\n')
        standardized_smiles = standardize_smiles(smiles_list)
        valid_smiles = [smi for smi in standardized_smiles if smi is not None]
        if valid_smiles:
            # Dự đoán bằng XGBoost
            features = [smiles_to_features(smi) for smi in valid_smiles]
            xgb_model = load_xgb_model()
            xgb_predictions = xgb_model.predict(np.array(features))

            # Chuyển SMILES thành graph và dự đoán bằng GIN
            dataset = MyDataset(valid_smiles)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            gin_model = load_gin_model()
            gin_model.to(device)
            gin_predictions = []
            for batch in loader:
                batch = batch.to(device)
                with torch.no_grad():
                    pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                gin_predictions.extend(pred.cpu().numpy().flatten())

            # Tạo DataFrame cho kết quả
            result_df = pd.DataFrame({
                'SMILES đã chuẩn hóa': valid_smiles,
                'Dự đoán XGBoost': xgb_predictions,
                'Dự đoán pEC50 (GIN)': gin_predictions
            })
            st.write("Kết quả dự đoán:", result_df)
        else:
            st.write("Không có SMILES hợp lệ để dự đoán.")

else:
    uploaded_file = st.file_uploader("Tải lên file CSV chứa SMILES", type=["csv"])
    if uploaded_file and st.button("Dự đoán"):
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
            standardized_smiles = standardize_smiles(smiles_list)
            df['Standardized_SMILES'] = standardized_smiles
            valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
            if valid_indices:
                valid_smiles = [standardized_smiles[i] for i in valid_indices]

                # Dự đoán bằng XGBoost
                features = [smiles_to_features(smi) for smi in valid_smiles]
                xgb_model = load_xgb_model()
                xgb_predictions = xgb_model.predict(np.array(features))

                # Chuyển SMILES thành graph và dự đoán bằng GIN
                dataset = MyDataset(valid_smiles)
                loader = DataLoader(dataset, batch_size=32, shuffle=False)
                gin_model = load_gin_model()
                gin_model.to(device)
                gin_predictions = []
                for batch in loader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    gin_predictions.extend(pred.cpu().numpy().flatten())

                # Thêm cột dự đoán vào DataFrame
                df.loc[valid_indices, 'Prediction_XGBoost'] = xgb_predictions
                df.loc[valid_indices, 'Prediction_pEC50_GIN'] = gin_predictions
                st.write("Dữ liệu với dự đoán:", df[['SMILES', 'Standardized_SMILES', 'Prediction_XGBoost', 'Prediction_pEC50_GIN']])
            else:
                st.write("Không có SMILES hợp lệ để dự đoán.")
        else:
            st.write("File CSV phải chứa cột 'SMILES'")

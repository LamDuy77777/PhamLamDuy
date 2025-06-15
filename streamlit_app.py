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
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# Tắt thông báo lỗi từ RDKit
RDLogger.DisableLog('rdApp.*')

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

# Hàm chuyển đổi SMILES thành đặc trưng Morgan fingerprint cho XGBoost
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Hàm chuyển đổi SMILES thành dữ liệu PyTorch Geometric cho GIN
def smi_to_pyg(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    atom_featurizer = MultiHotAtomFeaturizer()
    bond_featurizer = MultiHotBondFeaturizer()
    x = torch.tensor(atom_featurizer(mol), dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_features = bond_featurizer(bond)
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Định nghĩa lớp Dataset cho GIN
class MyDataset(Dataset):
    def __init__(self, smiles_list):
        self.data_list = [smi_to_pyg(smi) for smi in smiles_list if smi_to_pyg(smi) is not None]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

# Định nghĩa mô hình GIN đơn giản (giả định)
class MyGIN(nn.Module):
    def __init__(self, node_dim=72, edge_dim=14, embedding_dim=256, num_layers=7):
        super(MyGIN, self).__init__()
        self.embedding = nn.Linear(node_dim, embedding_dim)
        self.layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embedding_dim, 1)
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding(x)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# Tải mô hình XGBoost
@st.cache_resource
def load_xgb_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Tải mô hình GIN
@st.cache_resource
def load_gin_model():
    model = MyGIN().to(device)
    with open('GIN_597_562_cpu.pkl', 'rb') as f:
        state_dict = torch.load(f, map_location=device)
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
        
        # Lọc SMILES hợp lệ
        valid_smiles = [smi for smi in standardized_smiles if smi is not None]
        if valid_smiles:
            # Dự đoán với mô hình XGBoost
            features = [smiles_to_features(smi) for smi in valid_smiles]
            xgb_model = load_xgb_model()
            xgb_predictions = xgb_model.predict(np.array(features))
            
            # Dự đoán với mô hình GIN
            dataset = MyDataset(valid_smiles)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            gin_model = load_gin_model()
            gin_predictions = []
            for batch in dataloader:
                batch = batch.to(device)
                with torch.no_grad():
                    pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                gin_predictions.extend(pred.cpu().numpy().flatten())
            
            # Lọc các chất có dự đoán XGBoost là 1
            filtered_indices = [i for i, pred in enumerate(xgb_predictions) if pred == 1]
            filtered_smiles = [valid_smiles[i] for i in filtered_indices]
            filtered_gin_predictions = [gin_predictions[i] for i in filtered_indices]
            
            # Tạo DataFrame cho kết quả
            result_df = pd.DataFrame({
                'SMILES đã chuẩn hóa': filtered_smiles,
                'Dự đoán XGBoost': [1] * len(filtered_smiles),
                'Dự đoán pEC50 (GIN)': filtered_gin_predictions
            })
            st.write("Kết quả dự đoán:", result_df)
            
            # Nút tải về
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Tải xuống kết quả dưới dạng CSV",
                data=csv,
                file_name="du_doan.csv",
                mime="text/csv"
            )
        else:
            st.write("Không có SMILES hợp lệ để dự đoán.")

else:
    uploaded_file = st.file_uploader("Tải lên file CSV chứa SMILES", type=["csv"])
    if uploaded_file and st.button("Dự đoán"):
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
            standardized_smiles = standardize_smiles(smiles_list)
            
            # Lọc SMILES hợp lệ
            valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
            if valid_indices:
                valid_smiles = [standardized_smiles[i] for i in valid_indices]
                
                # Dự đoán với mô hình XGBoost
                features = [smiles_to_features(smi) for smi in valid_smiles]
                xgb_model = load_xgb_model()
                xgb_predictions = xgb_model.predict(np.array(features))
                
                # Dự đoán với mô hình GIN
                dataset = MyDataset(valid_smiles)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
                gin_model = load_gin_model()
                gin_predictions = []
                for batch in dataloader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    gin_predictions.extend(pred.cpu().numpy().flatten())
                
                # Lọc các chất có dự đoán XGBoost là 1
                filtered_indices = [i for i, pred in enumerate(xgb_predictions) if pred == 1]
                filtered_smiles = [valid_smiles[i] for i in filtered_indices]
                filtered_gin_predictions = [gin_predictions[i] for i in filtered_indices]
                
                # Tạo DataFrame cho kết quả
                result_df = pd.DataFrame({
                    'SMILES gốc': [smiles_list[i] for i in valid_indices if xgb_predictions[i] == 1],
                    'SMILES đã chuẩn hóa': filtered_smiles,
                    'Dự đoán XGBoost': [1] * len(filtered_smiles),
                    'Dự đoán pEC50 (GIN)': filtered_gin_predictions
                })
                st.write("Kết quả dự đoán:", result_df)
                
                # Nút tải về
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Tải xuống kết quả dưới dạng CSV",
                    data=csv,
                    file_name="du_doan.csv",
                    mime="text/csv"
                )
            else:
                st.write("Không có SMILES hợp lệ để dự đoán.")
        else:
            st.write("File CSV phải chứa cột 'SMILES'")

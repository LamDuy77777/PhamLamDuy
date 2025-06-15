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
from torch_geometric.data import Data, Dataset
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer

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

# Tải mô hình XGBoost
@st.cache_resource
def load_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Giao diện ứng dụng
st.title("Dự đoán với Mô hình XGBoost")

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
            # Chuyển đổi SMILES thành đặc trưng
            features = [smiles_to_features(smi) for smi in valid_smiles]
            model = load_model()
            predictions = model.predict(np.array(features))
            
            # Tạo DataFrame cho kết quả
            result_df = pd.DataFrame({
                'SMILES đã chuẩn hóa': valid_smiles,
                'Dự đoán': predictions
            })
            st.write("Kết quả dự đoán:", result_df)
            
            # Tạo dataset PyG và lưu thành file .pkl
            dataset = MyDataset(valid_smiles)
            with open('dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)
            with open('dataset.pkl', 'rb') as f:
                st.download_button('Tải dataset PyG (.pkl)', f, file_name='dataset.pkl')
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
            
            # Lọc SMILES hợp lệ
            valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
            if valid_indices:
                valid_smiles = [standardized_smiles[i] for i in valid_indices]
                features = [smiles_to_features(smi) for smi in valid_smiles]
                model = load_model()
                predictions = model.predict(np.array(features))
                
                # Thêm cột dự đoán vào DataFrame
                df.loc[valid_indices, 'Prediction'] = predictions
                st.write("Dữ liệu với dự đoán:", df[['SMILES', 'Standardized_SMILES', 'Prediction']])
                
                # Tạo dataset PyG và lưu thành file .pkl
                dataset = MyDataset(valid_smiles)
                with open('dataset.pkl', 'wb') as f:
                    pickle.dump(dataset, f)
                with open('dataset.pkl', 'rb') as f:
                    st.download_button('Tải dataset PyG (.pkl)', f, file_name='dataset.pkl')
            else:
                st.write("Không có SMILES hợp lệ để dự đoán.")
        else:
            st.write("File CSV phải chứa cột 'SMILES'")

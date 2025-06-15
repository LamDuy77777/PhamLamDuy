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
                'STT': range(1, len(valid_smiles) + 1),
                'SMILES đã chuẩn hóa': valid_smiles,
                'Dự đoán': predictions
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
            
            # Lọc SMILES hợp lệ
            valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
            if valid_indices:
                valid_smiles = [standardized_smiles[i] for i in valid_indices]
                features = [smiles_to_features(smi) for smi in valid_smiles]
                model = load_model()
                predictions = model.predict(np.array(features))
                
                # Thêm cột dự đoán vào DataFrame
                df.loc[valid_indices, 'Prediction'] = predictions
                df['STT'] = range(1, len(df) + 1)
                st.write("Dữ liệu với dự đoán:", df[['STT', 'SMILES', 'Standardized_SMILES', 'Prediction']])
            else:
                st.write("Không có SMILES hợp lệ để dự đoán.")
        else:
            st.write("File CSV phải chứa cột 'SMILES'")

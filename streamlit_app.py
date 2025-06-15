import streamlit as st
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolStandardize

# Hàm chuẩn hóa SMILES từ code của bạn
invalid_smiles = []
def standardize(smiles, invalid_smiles_list):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        mol = Chem.RemoveHs(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.Reionize(mol)
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, kekuleSmiles=True)
        return standardized_smiles
    except Exception as e:
        print(f"Error standardizing SMILES {smiles}: {e}")
        invalid_smiles_list.append(smiles)
        return None

# Tải mô hình XGBoost
def load_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Giao diện Streamlit
st.title("Chuẩn hóa SMILES và Dự đoán bằng XGBoost")

# Chọn cách nhập SMILES
input_method = st.radio("Chọn cách nhập SMILES:", ("Nhập SMILES", "Tải lên tệp CSV"))

if input_method == "Nhập SMILES":
    smiles_input = st.text_input("Nhập SMILES")
    if st.button("Dự đoán"):
        if smiles_input:
            standardized_smiles = standardize(smiles_input, invalid_smiles)
            if standardized_smiles:
                # Giả định mô hình chấp nhận SMILES dạng chuỗi
                prediction = model.predict([standardized_smiles])
                st.write("SMILES đã chuẩn hóa:", standardized_smiles)
                st.write("Kết quả dự đoán:", prediction)
            else:
                st.write("SMILES không hợp lệ:", smiles_input)
        else:
            st.write("Vui lòng nhập SMILES")

elif input_method == "Tải lên tệp CSV":
    uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' in df.columns:
            standardized_smiles_list = []
            for smiles in df['SMILES']:
                standardized_smiles = standardize(smiles, invalid_smiles)
                standardized_smiles_list.append(standardized_smiles)
            df['Standardized_SMILES'] = standardized_smiles_list
            # Loại bỏ SMILES không hợp lệ để dự đoán
            valid_df = df.dropna(subset=['Standardized_SMILES'])
            if not valid_df.empty:
                predictions = model.predict(valid_df['Standardized_SMILES'].tolist())
                valid_df['Prediction'] = predictions
                st.write("Kết quả dự đoán:", valid_df[['SMILES', 'Standardized_SMILES', 'Prediction']])
            if invalid_smiles:
                st.write("Các SMILES không hợp lệ:", invalid_smiles)
        else:
            st.write("Tệp CSV phải có cột 'SMILES'")
    else:
        st.write("Vui lòng tải lên tệp CSV")

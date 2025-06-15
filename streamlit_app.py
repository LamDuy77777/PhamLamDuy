import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Draw
from rdkit import Chem, RDLogger


# Hàm tải mô hình
def load_classification_model():
    try:
        with open('classification_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

# Hàm chuẩn hóa SMILES
invalid_smiles = []
def standardize(smiles, invalid_smiles_list):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES không hợp lệ: {smiles}")
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standardized_smiles
    except Exception as e:
        invalid_smiles_list.append(smiles)
        st.error(f"Lỗi khi chuẩn hóa SMILES: {e}")
        return None

# Giao diện Streamlit
st.title("Chuẩn hóa SMILES và Dự đoán")

# Tải mô hình
model = load_classification_model()

if model:
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
else:
    st.write("Không thể tải mô hình. Vui lòng kiểm tra lại.")

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

# Tiêu đề và giới thiệu
st.title('🤖 Machine Learning App for pEC50 Prediction 🤖')
st.info('Ứng dụng này giúp bạn dự đoán pEC50 của các chất chủ vận thụ thể apelin bằng kỹ thuật học máy tiên tiến.')

# Phần dữ liệu
st.write('### Dữ liệu dùng để xây dựng mô hình')
with st.expander('Dữ liệu đã chuẩn hóa'):
    st.write('#### Dữ liệu Apelin đã chuẩn hóa')
    df = pd.read_csv('https://raw.githubusercontent.com/LamDuy77777/data/refs/heads/main/Apelin_1715.csv')
    st.dataframe(df)

# Phần trực quan hóa dữ liệu
with st.expander('Trực quan hóa dữ liệu'):
    st.write("### Phân bố của pEC50")
    st.bar_chart(df['pEC50'])  # Giả sử cột 'pEC50' có trong dữ liệu

# Thanh bên để nhập dữ liệu
st.sidebar.header('Nhập thông tin')
st.sidebar.write('Nhập chuỗi SMILES của hợp chất để dự đoán pEC50.')
smiles_input = st.sidebar.text_input('SMILES', '')
predict_button = st.sidebar.button('Dự đoán pEC50')

# Hàm chuyển đổi SMILES thành ECFP4
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

# Tải mô hình học máy đã huấn luyện từ file trên GitHub
try:
    # Giả sử file 'model.pkl' đã được tải lên repository của bạn
    with open('https://raw.githubusercontent.com/LamDuy77777/data/refs/heads/main/xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Không tìm thấy file mô hình 'model.pkl'. Vui lòng tải file lên repository.")
    st.stop()

# Logic dự đoán
if predict_button:
    if smiles_input:
        ecfp4 = smiles_to_ecfp4(smiles_input)
        if ecfp4 is not None:
            prediction = model.predict([ecfp4])
            st.success(f'pEC50 dự đoán: **{prediction[0]:.3f}**')
        else:
            st.error('Chuỗi SMILES không hợp lệ. Vui lòng kiểm tra và thử lại.')
    else:
        st.warning('Vui lòng nhập chuỗi SMILES để tiếp tục.')

# Thông tin bổ sung
st.write('### Cách hoạt động')
st.write('Từ các chuỗi SMILES đã chuẩn hóa, ứng dụng này tính toán dấu vân tay ECFP4 (2048 bit) và sử dụng mô hình học máy đã được huấn luyện sẵn để dự đoán giá trị pEC50.')

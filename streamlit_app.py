import streamlit as st
import pickle
# Giả định có hàm chuẩn hóa SMILES, thay bằng hàm thực tế của bạn
# from smiles_normalizer import normalize_smiles

# Tải mô hình với caching
@st.cache_resource
def load_classification_model():
    with open('classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_regression_model():
    with open('regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

classification_model = load_classification_model()
regression_model = load_regression_model()

# Khởi tạo session_state để theo dõi trang
if 'page' not in st.session_state:
    st.session_state.page = 'Giới thiệu'

# Nút điều hướng
col1, col2 = st.columns(2)
with col1:
    if st.button("Giới thiệu"):
        st.session_state.page = 'Giới thiệu'
with col2:
    if st.button("Dự đoán"):
        st.session_state.page = 'Dự đoán'

# Nội dung trang
if st.session_state.page == 'Giới thiệu':
    st.title("Dự đoán bằng AI")
    st.write("Chào mừng bạn đến với ứng dụng của tôi! Đây là nơi triển khai mô hình phân loại và hồi quy để dự đoán dựa trên SMILES.")
    st.write("Nhấn nút 'Dự đoán' để sử dụng các mô hình.")
elif st.session_state.page == 'Dự đoán':
    st.title("Dự đoán")
    model_type = st.selectbox("Chọn mô hình", ["Phân loại", "Hồi quy"])
    smiles_input = st.text_input("Nhập SMILES")
    if st.button("Dự đoán"):
        if smiles_input:
            try:
                # Chuẩn hóa SMILES (thay bằng hàm thực tế nếu có)
                normalized_smiles = smiles_input  # Placeholder
                st.write("SMILES đã chuẩn hóa:", normalized_smiles)
                if model_type == "Phân loại":
                    prediction = classification_model.predict([normalized_smiles])
                    st.write("Kết quả phân loại:", prediction)
                elif model_type == "Hồi quy":
                    prediction = regression_model.predict([normalized_smiles])
                    st.write("Kết quả hồi quy:", prediction)
            except Exception as e:
                st.write("Lỗi:", str(e))
        else:
            st.write("Vui lòng nhập SMILES")

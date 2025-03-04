import streamlit as st

# Khởi tạo session_state để theo dõi trang hiện tại
if 'page' not in st.session_state:
    st.session_state.page = 'Giới thiệu'

# Tạo ba nút chuyển trang ở đầu trang bằng cách dùng columns
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Giới thiệu"):
        st.session_state.page = 'Giới thiệu'
with col2:
    if st.button("Chuẩn hóa SMILES"):
        st.session_state.page = 'Chuẩn hóa SMILES'
with col3:
    if st.button("Tải mô hình"):
        st.session_state.page = 'Tải mô hình'

# Hiển thị nội dung dựa trên trang hiện tại
if st.session_state.page == 'Giới thiệu':
    st.title('**Machine Learning Model**')
    st.write("Chào mừng bạn đến với ứng dụng của tôi! Đây là nơi để giới thiệu về dự án và các chức năng chính.")
    st.write("Sử dụng các nút ở trên để chuyển sang các trang khác.")
    st.header("Sử dụng st.selectbox để tạo hộp chọn")
    color = st.selectbox("Em thích màu gì",("đen", "trắng", "xanh dương", "tím nhạt"))
    button = st.button("Submit answer")
    st.form_submit_button(label="Browse file", type="secondary", icon="🚨")
    if button:
        st.write(color)
    st.header("Sử dụng st.multiselect để tạo hộp multiselect")
    st.multiselect("Thích con gì dzay?", ["cá", "gà", "thỏ"])
    st.sidebar.header("Option")
    
    

elif st.session_state.page == 'Chuẩn hóa SMILES':
    st.title("Trang Chuẩn hóa SMILES")
    st.write("Tại đây, bạn có thể nhập chuỗi SMILES và chuẩn hóa nó.")
    smiles_input = st.text_input("Nhập chuỗi SMILES")
    if smiles_input:
        # Giả sử bạn sẽ thêm hàm chuẩn hóa thực tế sau
        st.write(f"Chuỗi SMILES đã chuẩn hóa: {smiles_input}")  # Thay bằng hàm thực tế nếu có

elif st.session_state.page == 'Tải mô hình':
    st.title("Trang Tải Mô hình")
    st.write("Tại đây, bạn có thể tải và xem thông tin về mô hình đã huấn luyện.")
    st.write("Chức năng tải mô hình sẽ được thêm sau khi bạn cung cấp chi tiết cụ thể.")

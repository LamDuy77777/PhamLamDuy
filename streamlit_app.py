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
    if button:
        st.write(color)
    st.header("Sử dụng st.multiselect để tạo hộp multiselect")
    st.multiselect("Thích con gì dzay?", ["cá", "gà", "thỏ"])
    st.sidebar.header("Option")
    
    

elif st.session_state.page == 'Chuẩn hóa SMILES':
    # Tiêu đề ứng dụng
    
    st.title("Trang Chuẩn hóa SMILES")
    # Tạo ô browse file
    uploaded_file = st.file_uploader("Chọn một file để tải lên", type=["txt", "csv", "pdf"])

    # Xử lý file sau khi được tải lên
    if uploaded_file is not None:
        # Hiển thị thông tin file
        st.write("Tên file:", uploaded_file.name)
        st.write("Kích thước file:", uploaded_file.size, "bytes")
    
        # Đọc nội dung file (ví dụ với file text)
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.write("Nội dung file:")
            st.text(content)
        else:
            st.write("File đã được tải lên, nhưng chưa xử lý nội dung.")
    else:
        st.write("Vui lòng tải lên một file!")
elif st.session_state.page == 'Tải mô hình':
    st.title("Trang Tải Mô hình")
    st.write("Tại đây, bạn có thể tải và xem thông tin về mô hình đã huấn luyện.")
    st.write("Chức năng tải mô hình sẽ được thêm sau khi bạn cung cấp chi tiết cụ thể.")

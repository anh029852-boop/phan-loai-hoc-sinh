import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cấu hình trang
st.set_page_config(page_title="Phân tích Phiên học tập", layout="wide")

def run_clustering(data_list):
    if len(data_list) < 3:
        return None, "Cần ít nhất 3 phiên học để tiến hành phân cụm."

    # Chuyển dữ liệu sang DataFrame
    df = pd.DataFrame(data_list, columns=['focus_time', 'distractions', 'delay_start'])
    
    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Gán nhãn logic dựa trên centroids
    centroids = df.groupby('cluster').mean()
    label_map = {}
    
    # Xác định cụm theo đặc điểm
    c_idx_focus = centroids['focus_time'].idxmax()
    c_idx_delay = centroids['delay_start'].idxmax()
    
    label_map[c_idx_focus] = "Chất lượng"
    label_map[c_idx_delay] = "Trì hoãn"
    
    for i in range(3):
        if i not in label_map:
            label_map[i] = "Xao nhãng"
    
    df['category'] = df['cluster'].map(label_map)
    return df, None

def main():
    st.title("📊 Phân loại Phiên làm việc của Học sinh")
    st.markdown("""
    Ứng dụng này sử dụng học máy (K-Means) để phân loại các phiên học tập thành 3 nhóm: 
    **Chất lượng**, **Xao nhãng**, và **Trì hoãn**.
    """)

    # Khởi tạo state để lưu dữ liệu nếu chưa có
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    # Bố cục cột
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Nhập dữ liệu")
        with st.form("input_form", clear_on_submit=True):
            focus = st.number_input("Thời gian tập trung (phút)", min_value=0.0, step=1.0)
            distractions = st.number_input("Số lần xao nhãng", min_value=0, step=1)
            delay = st.number_input("Bắt đầu trễ (phút)", min_value=0.0, step=1.0)
            
            submit = st.form_submit_button("Thêm phiên học")
            if submit:
                st.session_state.sessions.append([focus, distractions, delay])
                st.success("Đã thêm!")

        if st.button("Xóa tất cả dữ liệu"):
            st.session_state.sessions = []
            st.rerun()

    with col2:
        st.header("Kết quả phân tích")
        if st.session_state.sessions:
            df_display = pd.DataFrame(st.session_state.sessions, columns=['Tập trung', 'Xao nhãng', 'Trì hoãn'])
            st.dataframe(df_display, use_container_width=True)
            
            if st.button("🚀 Tiến hành Phân cụm", type="primary"):
                df_result, error = run_clustering(st.session_state.sessions)
                
                if error:
                    st.error(error)
                else:
                    st.divider()
                    st.subheader("Bảng phân loại chi tiết")
                    st.dataframe(df_result[['focus_time', 'distractions', 'delay_start', 'category']], use_container_width=True)

                    # Trực quan hóa
                    st.subheader("Biểu đồ phân cụm 3D")
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    colors = {'Chất lượng': 'green', 'Xao nhãng': 'orange', 'Trì hoãn': 'red'}
                    
                    for cat, color in colors.items():
                        subset = df_result[df_result['category'] == cat]
                        if not subset.empty:
                            ax.scatter(subset['focus_time'], subset['distractions'], subset['delay_start'], 
                                       c=color, label=cat, s=100, alpha=0.7)
                    
                    ax.set_xlabel('Tập trung')
                    ax.set_ylabel('Xao nhãng')
                    ax.set_zlabel('Trì hoãn')
                    ax.legend()
                    
                    st.pyplot(fig)
        else:
            st.info("Chưa có dữ liệu. Vui lòng nhập thông tin ở cột bên trái.")

if __name__ == "__main__":
    main()

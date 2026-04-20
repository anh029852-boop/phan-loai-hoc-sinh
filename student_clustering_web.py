import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. THIẾT KẾ GIAO DIỆN (MODERN PINK THEME)
# ==========================================
st.set_page_config(page_title="AI Pink Focus - Phân loại học tập", layout="wide")

st.markdown("""
<style>
    /* Nền ứng dụng màu hồng cực nhạt */
    .stApp {
        background-color: #fff1f2; 
        color: #4c0519;
    }
    
    /* Tiêu đề chính màu hồng đậm */
    h1, h2, h3 {
        color: #be185d !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Tùy chỉnh Nút bấm hiện đại */
    .stButton>button {
        background-color: #fb7185;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #e11d48;
        transform: scale(1.02);
        color: white;
    }

    /* Khung nhập liệu trắng bo tròn */
    .stForm {
        background-color: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 2px solid #fecdd3;
    }

    /* Bảng dữ liệu */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOGIC: PHÂN LOẠI & LỜI KHUYÊN
# ==========================================

def get_detailed_advice(screen_time, notifications):
    # Logic phân loại dựa trên luật (Rule-based)
    if screen_time < 10 and notifications <= 1:
        return {
            "group": "Nhóm A: Chiến binh Deep Work (Focus Masters)",
            "analysis": "Screen Time cực thấp, thông báo bằng 0. Bạn đã làm chủ hoàn toàn môi trường học tập.",
            "status": "✨ Đây là nhóm 'mẫu hình'. Thời gian Deep Work chiếm trên 80% thời lượng phiên học.",
            "advice": "Lời khuyên: Duy trì phong độ này! Bạn đang sở hữu siêu năng lực tập trung trong kỷ nguyên số.",
            "color": "#15803d" # Xanh lá
        }
    elif notifications > 15 or screen_time > 40:
        return {
            "group": "Nhóm C: Bẫy xao nhãng (Digital Distraction Trap)",
            "analysis": "Thông báo quá cao dẫn đến Screen Time chiếm phần lớn phiên học.",
            "status": "🔴 Deep Work Time gần như bằng 0. Bạn đang bị thuật toán giải trí dẫn dắt hoàn toàn.",
            "advice": "Lời khuyên: AI đề xuất cảnh báo đỏ! Hãy sử dụng AppBlock để khóa ứng dụng và bật chế độ Do Not Disturb cưỡng bách.",
            "color": "#b91c1c" # Đỏ
        }
    else:
        return {
            "group": "Nhóm B: Vùng dao động (The Shallow Workers)",
            "analysis": "Screen Time và thông báo ở mức trung bình. Có nỗ lực nhưng chưa quyết liệt.",
            "status": "🔶 Trạng thái 'Học nông' (Shallow Work). Bạn vẫn làm được bài nhưng không có sự sáng tạo đột phá.",
            "advice": "Lời khuyên: Hãy thử phương pháp Pomodoro và cất điện thoại sang phòng khác để chuyển dịch sang Nhóm A.",
            "color": "#b45309" # Cam
        }

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================

def main():
    st.title("🌸 AI Pink Focus: Phân cụm K-Means & Lời khuyên")
    st.write("Nhập dữ liệu phiên học của bạn để AI tiến hành phân tích chuyên sâu.")

    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("📍 Nhập thông số")
        with st.form("input_form", clear_on_submit=True):
            focus = st.number_input("Tổng thời gian học (phút)", min_value=1.0, value=60.0)
            screen = st.number_input("Screen Time (phút)", min_value=0.0, value=5.0)
            notis = st.number_input("Số thông báo", min_value=0, step=1)
            distractions = st.number_input("Số lần xao nhãng khác", min_value=0, step=1)
            delay = st.number_input("Bắt đầu trễ (phút)", min_value=0.0)
            
            if st.form_submit_button("Lưu phiên học"):
                st.session_state.sessions.append({
                    'Focus': focus, 'ScreenTime': screen, 'Notis': notis, 
                    'Distractions': distractions, 'Delay': delay
                })
                st.success("Đã thêm dữ liệu!")

        if st.button("🗑️ Xóa dữ liệu"):
            st.session_state.sessions = []
            st.rerun()

    with col2:
        if st.session_state.sessions:
            df = pd.DataFrame(st.session_state.sessions)
            st.subheader("📋 Lịch sử phiên học")
            st.dataframe(df, use_container_width=True)

            if len(df) >= 3:
                if st.button("🚀 KÍCH HOẠT AI PHÂN CỤM (K-MEANS)", type="primary"):
                    st.divider()
                    
                    # Chạy K-Means
                    X = df[['Focus', 'ScreenTime', 'Notis', 'Distractions', 'Delay']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df['AI_Cluster'] = kmeans.fit_predict(X_scaled)

                    # Vẽ biểu đồ 3D
                    st.subheader("📊 Bản đồ phân cụm AI")
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    colors = ['#db2777', '#f472b6', '#9d174d'] # Các tông hồng
                    for i in range(3):
                        subset = df[df['AI_Cluster'] == i]
                        ax.scatter(subset['Focus'], subset['ScreenTime'], subset['Notis'], 
                                   s=100, label=f'Cụm AI {i}', c=colors[i])
                    
                    ax.set_xlabel('Tập trung')
                    ax.set_ylabel('Screen Time')
                    ax.set_zlabel('Thông báo')
                    st.pyplot(fig)

                    # ĐƯA RA LỜI KHUYÊN CHO PHIÊN CUỐI
                    st.divider()
                    st.subheader("💡 Nhận xét & Lời khuyên từ AI")
                    
                    last = df.iloc[-1]
                    advice_data = get_detailed_advice(last['ScreenTime'], last['Notis'])
                    
                    st.markdown(f"""
                    <div style="background-color: white; padding: 25px; border-left: 10px solid {advice_data['color']}; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <h3 style="margin-top:0; color: {advice_data['color']};">{advice_data['group']}</h3>
                        <p style="font-size: 1.1rem;"><b>Phân tích:</b> {advice_data['analysis']}</p>
                        <p style="font-size: 1.1rem; color: #333;"><b>Trạng thái:</b> {advice_data['status']}</p>
                        <hr style="border: 0.5px solid #eee;">
                        <p style="font-size: 1.2rem; font-weight: bold; color: {advice_data['color']};">✨ {advice_data['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"Thông tin thêm: Thuật toán K-Means đã xếp phiên này vào nhóm tương đồng số {last['AI_Cluster']}.")
            else:
                st.warning("⚠️ Vui lòng nhập ít nhất 3 phiên học để AI có đủ dữ liệu phân cụm.")
        else:
            st.info("Chưa có dữ liệu. Hãy nhập ở cột bên trái.")

    st.markdown("<br><br><p style='text-align: center; color: #fb7185;'>💖 Học tập hiệu quả hơn mỗi ngày cùng AI Pink Focus 💖</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

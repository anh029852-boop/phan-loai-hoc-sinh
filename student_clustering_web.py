import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (THEME & STYLE)
# ==========================================
st.set_page_config(page_title="AI Focus: Xếp loại xao nhãng học sinh", layout="wide")

# CSS tùy chỉnh để chữ trắng nổi bật trên nền hồng
st.markdown("""
<style>
    /* Nền tổng thể màu hồng trung tính */
    .stApp {
        background-color: #f472b6; 
        color: #ffffff;
    }
    
    /* Tiêu đề chính chữ trắng, in đậm, có bóng đổ để rõ nét */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    /* Chữ trong toàn bộ ứng dụng */
    p, label, .stMarkdown {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Các phần quan trọng cần in đậm */
    b, strong {
        font-weight: 900 !important;
        color: #fff1f2 !important;
        text-decoration: underline rgba(255,255,255,0.3);
    }

    /* Tùy chỉnh ô nhập liệu (Input) để không bị trùng màu */
    .stNumberInput div div input {
        background-color: #ffffff !important;
        color: #be185d !important;
        font-weight: bold !important;
        border-radius: 10px;
    }

    /* Tùy chỉnh Nút bấm */
    .stButton>button {
        background-color: #9d174d;
        color: #ffffff !important;
        border-radius: 15px;
        border: 2px solid #ffffff;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #9d174d !important;
    }

    /* Bảng dữ liệu */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px;
    }

    /* Khung nhận xét */
    .comment-box {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ffffff;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC PHÂN LOẠI & NHẬN XÉT
# ==========================================

def get_advice_styled(category):
    if "A" in category:
        return {
            "title": "NHÓM A: CHIẾN BINH DEEP WORK (FOCUS MASTERS)",
            "status": "Trạng thái: **Làm chủ hoàn toàn môi trường học tập.**",
            "analysis": "Đặc điểm: **Screen Time cực thấp (< 10 phút/phiên)**, số thông báo gần như bằng 0. Thời gian Deep Work chiếm **trên 80%**.",
            "advice": "💡 Lời khuyên: Đây là mẫu hình tối ưu. Hãy tiếp tục duy trì và bảo vệ không gian tập trung này!",
            "border": "#22c55e"
        }
    elif "C" in category:
        return {
            "title": "NHÓM C: BẪY XAO NHÃNG (DIGITAL DISTRACTION TRAP)",
            "status": "Trạng thái: **Đang bị thuật toán giải trí dẫn dắt hoàn toàn.**",
            "analysis": "Đặc điểm: **Thông báo cao (> 15)** dẫn đến **Screen Time chiếm phần lớn (> 40 phút)**. Deep Work gần như bằng 0.",
            "advice": "⚠️ Cảnh báo: AI đề xuất kịch bản thay đổi hành vi cưỡng bách. Hãy khóa ứng dụng hoàn toàn trong giờ học!",
            "border": "#ef4444"
        }
    else:
        return {
            "title": "NHÓM B: VÙNG DAO ĐỘNG (THE SHALLOW WORKERS)",
            "status": "Trạng thái: **Học nông (Shallow Work), chưa bứt phá.**",
            "analysis": "Đặc điểm: **Screen Time và Thông báo mức trung bình**. Có nỗ lực nhưng chưa quyết liệt loại bỏ xao nhãng.",
            "advice": "🔶 Lời khuyên: Bạn cần quyết liệt hơn. Thử cất điện thoại sang phòng khác để chuyển từ 'học nông' sang 'học sâu'.",
            "border": "#f59e0b"
        }

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================

def main():
    st.title("🚀 AI Focus: Xếp loại xao nhãng học sinh")
    st.write("---")

    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("📥 Nhập dữ liệu phiên")
        with st.form("input_form", clear_on_submit=True):
            focus = st.number_input("Tổng thời gian phiên (phút)", min_value=1.0, value=60.0)
            screen = st.number_input("Screen Time (phút)", min_value=0.0, value=5.0)
            notis = st.number_input("Số lượng thông báo", min_value=0, step=1)
            distractions = st.number_input("Lần xao nhãng khác", min_value=0, step=1)
            
            if st.form_submit_button("LƯU PHIÊN HỌC"):
                if screen < 10 and notis <= 1:
                    cat = "Nhóm A"
                elif notis > 15 or screen > 40:
                    cat = "Nhóm C"
                else:
                    cat = "Nhóm B"
                
                st.session_state.sessions.append({
                    'Focus': focus, 'ScreenTime': screen, 'Notis': notis, 
                    'Distractions': distractions, 'Category': cat
                })
                st.rerun()

        if st.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU"):
            st.session_state.sessions = []
            st.rerun()

    with col2:
        if st.session_state.sessions:
            df = pd.DataFrame(st.session_state.sessions)
            st.subheader("📋 Danh sách dữ liệu")
            # Hiển thị bảng với màu sắc rõ ràng
            st.dataframe(df[['Focus', 'ScreenTime', 'Notis', 'Category']], use_container_width=True)

            if len(df) >= 3:
                if st.button("🚀 KÍCH HOẠT AI CLUSTERING (K-MEANS)"):
                    st.divider()
                    
                    # Học máy K-Means
                    X = df[['Focus', 'ScreenTime', 'Notis', 'Distractions']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df['AI_Cluster'] = kmeans.fit_predict(X_scaled)

                    # Biểu đồ 3D
                    st.subheader("🌐 Không gian xao nhãng (AI Clustering)")
                    fig = plt.figure(figsize=(8, 5), facecolor='#f472b6')
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_facecolor('#f472b6')
                    
                    # Màu sắc các cụm (Trắng, Đen, Hồng đậm) để nổi bật trên nền
                    cluster_colors = ['#ffffff', '#000000', '#9d174d']
                    for i in range(len(np.unique(df['AI_Cluster']))):
                        c_data = df[df['AI_Cluster'] == i]
                        ax.scatter(c_data['Focus'], c_data['ScreenTime'], c_data['Notis'], 
                                   s=150, c=cluster_colors[i], edgecolors='white', label=f'Cụm {i}')
                    
                    ax.set_xlabel('Tập trung', color='white', fontweight='bold')
                    ax.set_ylabel('Screen Time', color='white', fontweight='bold')
                    ax.set_zlabel('Thông báo', color='white', fontweight='bold')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.tick_params(axis='z', colors='white')
                    st.pyplot(fig)

                    # Hiển thị Lời khuyên cho phiên cuối
                    st.divider()
                    last_session = df.iloc[-1]
                    advice = get_advice_styled(last_session['Category'])
                    
                    st.markdown(f"""
                    <div class="comment-box" style="border-left: 10px solid {advice['border']};">
                        <h2 style="margin-top:0;">{advice['title']}</h2>
                        <p style="font-size: 1.1rem;">{advice['status']}</p>
                        <p style="font-size: 1rem;">{advice['analysis']}</p>
                        <hr style="border-color: rgba(255,255,255,0.2);">
                        <p style="font-size: 1.2rem; font-weight: bold;">{advice['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"ℹ️ *AI ghi chú: Phiên này thuộc Cụm số **{last_session['AI_Cluster']}** dựa trên phân tích tương đồng dữ liệu.*")
            else:
                st.warning("⚠️ Cần ít nhất 3 phiên học để AI bắt đầu phân cụm K-Means.")
        else:
            st.info("Chưa có dữ liệu. Vui lòng nhập thông số ở cột bên trái.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN THEO PALETTE ẢNH 2
# ==========================================
st.set_page_config(page_title="AI Focus: Xếp loại xao nhãng học sinh", layout="wide")

# Các mã màu từ palette ảnh 2
SAVORY_SAGE = "#818263"
AVOCADO_SMOOTHIE = "#C2C395"
BLUSH_BEET = "#DDBAAE"
PEACH_PROTEIN = "#EFD7CF"
OAT_LATTE = "#DCD4C1"
HONEY_OATMILK = "#F6EAD4"

st.markdown(f"""
<style>
    /* Nền màu hồng pastel nhạt (Peach Protein) */
    .stApp {{
        background-color: {PEACH_PROTEIN};
        color: {SAVORY_SAGE};
    }}
    
    /* Tiêu đề chính dùng màu đậm nhất (Savory Sage) */
    h1 {{
        color: {SAVORY_SAGE} !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 30px;
    }}
    
    h2, h3 {{
        color: {SAVORY_SAGE} !important;
        font-weight: 700 !important;
    }}

    /* Chữ văn bản chính in đậm và màu sẫm để rõ nét */
    p, label, .stMarkdown {{
        color: {SAVORY_SAGE} !important;
        font-weight: 600 !important;
    }}

    /* Tùy chỉnh ô nhập liệu */
    .stNumberInput div div input {{
        background-color: #ffffff !important;
        color: {SAVORY_SAGE} !important;
        font-weight: bold !important;
        border: 2px solid {BLUSH_BEET};
    }}

    /* Nút bấm */
    .stButton>button {{
        background-color: {SAVORY_SAGE};
        color: #ffffff !important;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {AVOCADO_SMOOTHIE};
        color: {SAVORY_SAGE} !important;
    }}

    /* Khung nhận xét */
    .comment-box {{
        background-color: {OAT_LATTE};
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid {SAVORY_SAGE};
        box-shadow: 2px 5px 15px rgba(0,0,0,0.05);
    }}
    
    /* Form nhập liệu */
    .stForm {{
        background-color: {HONEY_OATMILK};
        border: 2px solid {BLUSH_BEET};
        border-radius: 20px;
        padding: 20px;
    }}

    .stDataFrame {{
        background-color: white !important;
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HÀM PHÂN LOẠI & NHẬN XÉT
# ==========================================

def get_advice_styled(category, deep_work_ratio):
    if "A" in category:
        return {
            "title": "NHÓM A: CHIẾN BINH DEEP WORK",
            "status": "Trạng thái: *Làm chủ môi trường học tập.*",
            "analysis": f"Đặc điểm: Screen Time cực thấp. Thời lượng học tập trung chiếm **{deep_work_ratio:.1f}%** tổng phiên.",
            "advice": "💡 Lời khuyên: Bạn là mẫu hình lý tưởng! AI nhận diện đây là trạng thái tối ưu để các nhóm khác học hỏi.",
            "color": SAVORY_SAGE
        }
    elif "C" in category:
        return {
            "title": "NHÓM C: BẪY XAO NHÃNG",
            "status": "Trạng thái: *Bị thuật toán dẫn dắt hoàn toàn.*",
            "analysis": f"Đặc điểm: Thông báo > 15. Tỷ lệ tập trung thực tế chỉ đạt **{deep_work_ratio:.1f}%**.",
            "advice": "⚠️ Cảnh báo: Cần kịch bản thay đổi hành vi cưỡng bách (khóa ứng dụng) để thoát khỏi bẫy xao nhãng.",
            "color": "#944d4d"
        }
    else:
        return {
            "title": "NHÓM B: VÙNG DAO ĐỘNG",
            "status": "Trạng thái: *Học nông (Shallow Work).*",
            "analysis": f"Đặc điểm: Dao động giữa tập trung và xao nhãng. Tỷ lệ học sâu đạt **{deep_work_ratio:.1f}%**.",
            "advice": "🔶 Lời khuyên: Bạn chưa quyết liệt loại bỏ tác nhân gây nhiễu. Hãy nâng cao kỷ luật để chuyển dịch sang Nhóm A.",
            "color": "#947b4d"
        }

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================

def main():
    st.markdown("<h1>🚀 AI Focus: Xếp loại xao nhãng học sinh</h1>", unsafe_allow_html=True)

    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("📥 Nhập dữ liệu")
        with st.form("input_form", clear_on_submit=True):
            focus_total = st.number_input("Tổng thời gian phiên (phút)", min_value=1.0, value=60.0)
            # PHẦN MỚI THÊM: THỜI LƯỢNG HỌC TẬP TRUNG
            deep_work = st.number_input("Thời lượng học tập trung (phút)", min_value=0.0, value=45.0)
            
            screen = st.number_input("Screen Time (phút)", min_value=0.0, value=5.0)
            notis = st.number_input("Số thông báo nhận được", min_value=0, step=1)
            distractions = st.number_input("Số lần xao nhãng khác", min_value=0, step=1)
            
            if st.form_submit_button("LƯU PHIÊN HỌC"):
                if deep_work > focus_total:
                    st.error("Lỗi: Thời gian học tập trung không thể lớn hơn tổng thời gian.")
                else:
                    # Phân loại dựa trên quy tắc bạn đưa ra
                    if screen < 10 and notis <= 1:
                        cat = "Nhóm A"
                    elif notis > 15 or screen > 40:
                        cat = "Nhóm C"
                    else:
                        cat = "Nhóm B"
                    
                    st.session_state.sessions.append({
                        'Total': focus_total, 
                        'DeepWork': deep_work,
                        'ScreenTime': screen, 
                        'Notis': notis, 
                        'Distractions': distractions, 
                        'Category': cat
                    })
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU"):
            st.session_state.sessions = []
            st.rerun()

    with col2:
        if st.session_state.sessions:
            df = pd.DataFrame(st.session_state.sessions)
            st.subheader("📋 Lịch sử dữ liệu")
            # Hiển thị DataFrame rõ ràng
            st.dataframe(df[['Total', 'DeepWork', 'ScreenTime', 'Notis', 'Category']], use_container_width=True)

            if len(df) >= 3:
                if st.button("🚀 KÍCH HOẠT AI CLUSTERING (K-MEANS)"):
                    st.divider()
                    
                    # Chuẩn bị dữ liệu cho K-Means (bao gồm cả DeepWork)
                    X = df[['Total', 'DeepWork', 'ScreenTime', 'Notis', 'Distractions']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df['AI_Cluster'] = kmeans.fit_predict(X_scaled)

                    # Biểu đồ 3D
                    st.subheader("🌐 Không gian phân loại học tập")
                    fig = plt.figure(figsize=(8, 5), facecolor=PEACH_PROTEIN)
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_facecolor(HONEY_OATMILK)
                    
                    cluster_colors = [SAVORY_SAGE, AVOCADO_SMOOTHIE, BLUSH_BEET]
                    for i in range(len(np.unique(df['AI_Cluster']))):
                        c_data = df[df['AI_Cluster'] == i]
                        # Trục: DeepWork, ScreenTime, Notis
                        ax.scatter(c_data['DeepWork'], c_data['ScreenTime'], c_data['Notis'], 
                                   s=150, c=cluster_colors[i], edgecolors=SAVORY_SAGE, label=f'Cụm {i}')
                    
                    ax.set_xlabel('Học tập trung', color=SAVORY_SAGE, fontweight='bold')
                    ax.set_ylabel('Screen Time', color=SAVORY_SAGE, fontweight='bold')
                    ax.set_zlabel('Thông báo', color=SAVORY_SAGE, fontweight='bold')
                    st.pyplot(fig)

                    # Nhận xét phiên cuối
                    st.divider()
                    last_session = df.iloc[-1]
                    ratio = (last_session['DeepWork'] / last_session['Total']) * 100
                    advice = get_advice_styled(last_session['Category'], ratio)
                    
                    st.markdown(f"""
                    <div class="comment-box">
                        <h2 style="margin-top:0; color: {advice['color']} !important;">{advice['title']}</h2>
                        <p>{advice['status']}</p>
                        <p>{advice['analysis']}</p>
                        <hr style="border-color: {BLUSH_BEET};">
                        <p style="font-size: 1.15rem; font-weight: 800; color: {SAVORY_SAGE} !important;">{advice['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("ℹ️ Cần thêm dữ liệu (tối thiểu 3 phiên) để AI tiến hành phân cụm K-Means.")
        else:
            st.info("Chào bạn! Hãy nhập dữ liệu ở cột bên trái để bắt đầu phân tích.")

if __name__ == "__main__":
    main()

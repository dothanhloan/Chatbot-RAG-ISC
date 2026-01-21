import streamlit as st
import requests

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(page_title="ICS Chatbot", page_icon="🛡️")

st.title("🛡️ Trợ lý ảo ICS Security")
st.markdown("""
Chào mừng! Tôi là trợ lý AI của **Công ty Cổ phần An ninh Mạng Quốc tế (ICS)**.
Hãy hỏi tôi về:
- Giải pháp bảo mật **VietGuard** (Mobile Security)
- Hệ thống giám sát **AI SOC**
- Tiêu chuẩn **ISO 27001** và quy trình vận hành.
""")

# 2. KHỞI TẠO LỊCH SỬ CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. XỬ LÝ KHI NGƯỜI DÙNG NHẬP CÂU HỎI
if prompt := st.chat_input("Nhập câu hỏi của bạn về ICS..."):
    # Hiện câu hỏi của người dùng lên màn hình
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gửi câu hỏi sang Server API (Backend)
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu..."):
            try:
                # Gọi vào API mà bạn đang chạy ở cửa sổ cũ
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"question": prompt}
                )
                
                if response.status_code == 200:
                    ans = response.json().get("answer", "Lỗi: Không lấy được câu trả lời.")
                else:
                    ans = f"Lỗi Server: {response.status_code}"
            except Exception as e:
                ans = "⚠️ Lỗi kết nối: Hãy kiểm tra xem cửa sổ uvicorn (API) có đang chạy không!"

            st.markdown(ans)
    
    # Lưu câu trả lời của Bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": ans})
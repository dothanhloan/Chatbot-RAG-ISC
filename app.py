import streamlit as st
import os

# --- 1. CẤU HÌNH API KEY (QUAN TRỌNG NHẤT) ---
# Đoạn này giúp tự động lấy Key từ "Secrets" (nếu trên Web) hoặc ".env" (nếu dưới máy)
if "GROQ_API_KEY" in st.secrets:
    # Nếu chạy trên Streamlit Cloud -> Lấy từ Secrets
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    # Nếu chạy dưới máy Local -> Lấy từ file .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Kiểm tra lần cuối, nếu vẫn không có Key thì dừng lại báo lỗi
if not os.environ.get("GROQ_API_KEY"):
    st.error("❌ Lỗi: Chưa tìm thấy GROQ_API_KEY! Hãy cấu hình trong file .env (Local) hoặc mục Secrets (Cloud).")
    st.stop()

# --- 2. IMPORT THƯ VIỆN ---
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 3. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="ICS Chatbot", page_icon="🛡️")
st.title("🛡️ Trợ lý ảo ICS Security")
st.markdown("Hỏi đáp về giải pháp bảo mật **VietGuard**, **AI SOC** và tiêu chuẩn **ISO 27001** của ICS.")

# --- 4. HÀM NẠP DỮ LIỆU (CACHE ĐỂ KHÔNG PHẢI LOAD LẠI) ---
@st.cache_resource
def load_and_process_data():
    # Kiểm tra xem file có tồn tại không. 
    # Lưu ý: Theo cấu trúc GitHub của bạn [1], file có thể nằm trong thư mục 'data/' hoặc cùng cấp.
    # Code này sẽ thử tìm cả 2 nơi.
    file_path = "input.docx"
    if not os.path.exists(file_path):
        file_path = "data/input.docx" # Thử tìm trong thư mục data
        if not os.path.exists(file_path):
            return None

    # Đọc tài liệu
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    
    # Cắt nhỏ văn bản để AI dễ đọc
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Tạo Vector Database (Bộ nhớ)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- 5. KHỞI TẠO HỆ THỐNG ---
with st.spinner("Đang khởi động hệ thống tri thức ICS..."):
    vectorstore = load_and_process_data()

if vectorstore is None:
    st.error("⚠️ Không tìm thấy file 'input.docx'. Vui lòng kiểm tra lại thư mục dự án!")
else:
    # Cấu hình "Bộ não" AI (Llama 3 trên Groq)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    
    # Tạo khuôn mẫu câu trả lời chuyên nghiệp
    template = """
    Bạn là trợ lý AI chuyên nghiệp của Công ty Cổ phần An ninh Mạng Quốc tế (ICS).
    Sử dụng thông tin ngữ cảnh dưới đây để trả lời câu hỏi của khách hàng.
    Nếu thông tin không có trong ngữ cảnh, hãy nói là bạn chưa rõ, đừng bịa đặt.
    
    NGỮ CẢNH (Thông tin nội bộ ICS):
    {context}
    
    CÂU HỎI:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # --- 6. GIAO DIỆN CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Xử lý khi người dùng nhập câu hỏi
    if question := st.chat_input("Nhập câu hỏi về ICS (VD: VietGuard là gì?)..."):
        # Hiện câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # AI suy nghĩ và trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu dữ liệu..."):
                try:
                    # 1. Tìm kiếm thông tin liên quan trong input.docx
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(question)
                    context_text = "\n\n".join([d.page_content for d in relevant_docs])
                    
                    # 2. Gửi cho AI tổng hợp
                    chain = prompt | llm
                    response = chain.invoke({"context": context_text, "question": question})
                    
                    st.markdown(response.content)
                    
                    # Lưu câu trả lời
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {str(e)}")

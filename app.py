import streamlit as st
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="ICS Security Assistant",
    page_icon="🛡️",
    layout="wide"  # Chế độ xem rộng
)

# --- 2. TÙY CHỈNH GIAO DIỆN (CSS) ---
st.markdown("""
<style>
    h1 { color: #004d99; text-align: center; }
    .stChatMessage.st-emotion-cache-1c7y2kd { background-color: #f0f2f6; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. XỬ LÝ API KEY TỰ ĐỘNG ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

if not os.environ.get("GROQ_API_KEY"):
    st.error("❌ Lỗi: Chưa có API Key. Vui lòng kiểm tra cấu hình!")
    st.stop()

# --- 4. THANH BÊN (SIDEBAR) - THÔNG TIN ICS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9676/9676572.png", width=100)
    st.title("ICS Security")
    st.markdown("---")
    st.info("""
    🏢 **Thành lập:** 03/2020
    🏆 **Tiêu chuẩn:** ISO 27001
    🚀 **Sản phẩm:** VietGuard, AI SOC
    """)
    st.markdown("---")
    st.link_button("🌐 Website: icss.com.vn", "https://icss.com.vn")
    st.caption("© 2024 ICS Security")

# --- 5. NẠP DỮ LIỆU ---
@st.cache_resource
def load_data():
    # Tìm file input.docx trong thư mục
    possible_paths = ["input.docx", "data/input.docx"]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not file_path:
        return None

    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_data()

# --- 6. GIAO DIỆN CHAT CHÍNH ---
st.title("🛡️ Trợ lý Ảo An ninh Mạng ICS")
st.markdown("<p style='text-align: center;'>Hỗ trợ thông tin về <b>VietGuard</b>, <b>AI SOC</b> và chính sách bảo mật.</p>", unsafe_allow_html=True)

if vectorstore is None:
    st.warning("⚠️ Chưa tìm thấy file dữ liệu input.docx!")
else:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    template = """Bạn là trợ lý AI của công ty ICS. Dựa vào ngữ cảnh sau:
    {context}
    
    Hãy trả lời câu hỏi: {question}
    Trả lời ngắn gọn, chuyên nghiệp và thân thiện."""
    prompt = ChatPromptTemplate.from_template(template)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì về các giải pháp của ICS?"}]

    for msg in st.session_state.messages:
        icon = "🛡️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=icon):
            st.markdown(msg["content"])

    if question := st.chat_input("Nhập câu hỏi tại đây..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Đang tra cứu hệ thống..."):
                retriever = vectorstore.as_retriever()
                relevant = retriever.invoke(question)
                ctx = "\n".join([d.page_content for d in relevant])
                chain = prompt | llm
                res = chain.invoke({"context": ctx, "question": question})
                st.markdown(res.content)
        st.session_state.messages.append({"role": "assistant", "content": res.content})
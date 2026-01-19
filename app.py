import streamlit as st
import os
import sys

# ============================================
# CẤU HÌNH KEY (ĐIỀN ĐẦY ĐỦ 2 KEY)
# ============================================
KEY_GOOGLE_MOI = "AIzaSy_Dán_Key_Google_Mới_Vào_Đây"
KEY_GROQ_CUA_BAN = ""
# ============================================

os.environ["GOOGLE_API_KEY"] = KEY_GOOGLE_MOI
GROQ_API_KEY = KEY_GROQ_CUA_BAN

# --- KHẮC PHỤC LỖI FONT TRÊN WINDOWS ---
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # Quay lại dùng Google (Vì main.py của bạn đã chạy được nó!)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    st.error("❌ Thiếu thư viện!")
    st.stop()

st.set_page_config(page_title="Chatbot AI Tư Vấn", page_icon="🤖")
st.title("🤖 Chatbot AI Hỗ Trợ Tư Vấn")
st.write("---")

@st.cache_resource
def load_and_process_data():
    file_path = "data/input.docx"
    if not os.path.exists(file_path):
        return None
    
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    # 🛠️ CẤU HÌNH ĐẶC BIỆT ĐỂ KHÔNG BỊ LỖI ASCII
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=KEY_GOOGLE_MOI,
        transport="rest",       # Bắt buộc dùng REST
        client_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("⏳ Đang kết nối Google AI..."):
    try:
        vectorstore = load_and_process_data()
    except Exception as e:
        st.error(f"❌ Lỗi Google: {e}")
        st.stop()

if vectorstore is None:
    st.error("❌ Lỗi: Không tìm thấy file 'data/input.docx'")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

try:
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Lỗi Groq: {e}")
    st.stop()

# --- CHAT LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bạn cần hỏi gì?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        relevant_docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in relevant_docs])
        
        # Dùng Prompt tiếng Anh để Groq hiểu tốt hơn, nhưng yêu cầu trả lời tiếng Việt
        sys_prompt = ChatPromptTemplate.from_template(
            "Context: {context}\n\nQuestion: {question}\n\nAnswer in Vietnamese:"
        )
        
        chain = sys_prompt | llm
        response = chain.invoke({"context": context, "question": prompt})
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

    except Exception as e:
        st.error(f"Lỗi: {e}")
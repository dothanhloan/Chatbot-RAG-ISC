import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import các thư viện AI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Dùng thư viện mới chuẩn hơn
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. CẤU HÌNH
load_dotenv() 
sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(
    title="ICS Chatbot API",
    description="API cung cấp thông tin về giải pháp bảo mật VietGuard, AI SOC của ICS.",
    version="1.0"
)

# Biến toàn cục
vectorstore = None
llm = None

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# 2. KHỞI ĐỘNG SERVER (Load dữ liệu ICS từ input.docx)
@app.on_event("startup")
async def startup_event():
    global vectorstore, llm
    print("⏳ Đang khởi động hệ thống...")

    # A. Nạp dữ liệu từ input.docx
    file_path = "data/input.docx"
    if os.path.exists(file_path):
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        
        # Cắt nhỏ văn bản
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Tạo Vector (Dùng CPU để tránh lỗi DLL)
        print("🔄 Đang xử lý dữ liệu ICS...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Đã nạp dữ liệu ICS thành công!")
    else:
        print("❌ Cảnh báo: Không tìm thấy file data/input.docx")

    # B. Khởi tạo LLM (Điền Key trực tiếp ở đây để sửa lỗi)
    # HÃY DÁN KEY CỦA BẠN VÀO DƯỚI ĐÂY (Trong dấu ngoặc kép)
    api_key = "" 
    
    if not api_key or "gsk_" not in api_key:
        print("❌ Lỗi: Chưa điền API Key đúng trong file api.py")
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key
    )

# 3. ENDPOINT XỬ LÝ CHAT
@app.post("/chat", response_model=AnswerResponse)
async def chat_endpoint(request: QuestionRequest):
    global vectorstore, llm
    
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Dữ liệu chưa được nạp.")

    # Tìm kiếm thông tin liên quan
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(request.question)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt chuyên gia ICS (Dựa trên dữ liệu nguồn)
    template = """
    Bạn là trợ lý ảo của Công ty Cổ phần An ninh Mạng Quốc tế (ICS).
    
    THÔNG TIN CÔNG TY:
    - Thành lập: 3/2020. Trụ sở: TP.HCM & Hà Nội [1].
    - Sản phẩm: VietGuard (Mobile Security), Smart Dashboard, AI SOC [2].
    - Tiêu chuẩn: ISO 27001 [3].
    - Website: icss.com.vn [3].
    
    YÊU CẦU:
    Trả lời câu hỏi dựa trên ngữ cảnh (CONTEXT) bên dưới.
    Nếu không có thông tin, hãy nói: "Xin lỗi, tôi chỉ có thể hỗ trợ thông tin về các dịch vụ của ICS."
    
    CONTEXT:
    {context}
    
    CÂU HỎI:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": request.question})

    return AnswerResponse(answer=result.content)
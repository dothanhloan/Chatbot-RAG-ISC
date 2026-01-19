import os
import sys

# ==============================================================================
# 👇 KHU VỰC ĐIỀN CHÌA KHÓA (CẦN CẢ 2 CÁI)
# ==============================================================================

# 1. KEY GOOGLE MỚI (Để làm "Mắt" đọc tài liệu)
# 👉 Lấy tại: aistudio.google.com (Tạo Project mới cho sạch lỗi)
KEY_GOOGLE_MOI = ""

# 2. KEY GROQ (Để làm "Não" trả lời)
# 👉 Lấy tại: console.groq.com
KEY_GROQ_CUA_BAN = ""

# ==============================================================================

os.environ["GOOGLE_API_KEY"] = KEY_GOOGLE_MOI
GROQ_API_KEY = KEY_GROQ_CUA_BAN

try:
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_text_splitters import CharacterTextSplitter   # ✅ SỬA Ở ĐÂY
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print("❌ Thiếu thư viện hoặc xung đột môi trường:", e)
    print("👉 Chạy: py -3.12 -m pip install langchain langchain-groq langchain-community faiss-cpu docx2txt")
    sys.exit(1)


def main():
    file_path = "data/input.docx"
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file '{file_path}'")
        return

    print("📄 Đang đọc tài liệu...")
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)


    # ---------------------------------------------------------
    # 1. BỘ NHỚ (EMBEDDING) -> BẮT BUỘC DÙNG GOOGLE MODEL NÀY
    # ---------------------------------------------------------
    print("🧠 Đang nạp bộ nhớ (Google Embedding)...")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            # 👇 KHÔNG ĐƯỢC ĐỔI TÊN MODEL NÀY 👇
            model="models/text-embedding-004", 
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"❌ Lỗi Key Google: {e}")
        print("👉 Lời khuyên: Tạo Key Google mới tại aistudio.google.com rồi thay vào dòng số 9.")
        return

    # ---------------------------------------------------------
    # 2. BỘ NÃO (CHAT) -> DÙNG GROQ MODEL NÀY
    # ---------------------------------------------------------
    print("🔌 Đang kết nối não bộ Groq (Llama 3.3)...")
    try:
        llm = ChatGroq(
            temperature=0,
            # 👇 MODEL MỚI NHẤT CỦA GROQ 👇
            model_name="llama-3.3-70b-versatile", 
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        print(f"❌ Lỗi Key Groq: {e}")
        return

    prompt = ChatPromptTemplate.from_template(
        "Dựa vào văn bản: {context}\n\nTrả lời câu hỏi: {question}"
    )

    print("\n" + "="*40)
    print("🚀 CHATBOT GROQ (LLAMA 3.3) SẴN SÀNG!")
    print("="*40)

    while True:
        try:
            q = input("\n👤 Bạn: ")
            if q.lower() in ["exit", "thoát"]: break
            if not q.strip(): continue

            print("🤖 Bot: Đang suy nghĩ...", end="\r")
            
            relevant_docs = retriever.invoke(q)
            context = "\n".join([d.page_content for d in relevant_docs])
            res = (prompt | llm).invoke({"context": context, "question": q})
            print(f"\n💡 Trả lời: {res.content}")
            
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
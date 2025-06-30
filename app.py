import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import time

# --- CÁC HÀM XỬ LÝ ---

def get_text_from_files(files):
    """
    Đọc và trích xuất văn bản từ danh sách các tệp được tải lên (PDF, DOCX).
    """
    text = ""
    for file in files:
        # Lấy tên file để kiểm tra định dạng
        file_name = file.name
        st.write(f"Đang xử lý file: {file_name}")
        if file_name.endswith(".pdf"):
            try:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Lỗi khi đọc file PDF {file_name}: {e}")
        elif file_name.endswith(".docx"):
            try:
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Lỗi khi đọc file DOCX {file_name}: {e}")
    return text

def get_text_chunks(text):
    """
    Chia văn bản đầu vào thành các đoạn nhỏ hơn (chunks).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Chuyển đổi các đoạn văn bản thành vector và lưu trữ bằng FAISS.
    Sử dụng session state để lưu trữ, tránh tốn kém chi phí API.
    """
    try:
        # Khởi tạo mô hình embeddings của Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Tạo cơ sở dữ liệu vector từ các chunks
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Lưu vào session state để tái sử dụng
        st.session_state.vector_store = vector_store
        st.success("Đã vector hóa và lưu trữ tri thức thành công!")

    except Exception as e:
        st.error(f"Lỗi khi tạo vector store: {e}")
        st.error("Nguyên nhân có thể do API Key không hợp lệ hoặc chưa được cấp quyền. Hãy thử lại với key khác.")

def get_conversational_chain():
    """
    Tạo một chuỗi hỏi đáp (QA chain) với prompt tùy chỉnh.
    """
    prompt_template = """
    Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chi tiết và chính xác,
    dựa hoàn toàn vào nội dung trong "ngữ cảnh" được cung cấp.

    Hãy làm theo các quy tắc sau:
    1. Phân tích kỹ câu hỏi và ngữ cảnh trước khi trả lời.
    2. Trích xuất tất cả thông tin liên quan từ ngữ cảnh để xây dựng câu trả lời đầy đủ nhất.
    3. Nếu câu trả lời không có trong ngữ cảnh, hãy nói rõ: "Rất tiếc, tôi không tìm thấy thông tin trả lời cho câu hỏi này trong tài liệu bạn đã cung cấp."
    4. Tuyệt đối không được bịa đặt thông tin hoặc sử dụng kiến thức bên ngoài ngữ cảnh.
    5. Trình bày câu trả lời một cách rõ ràng, mạch lạc bằng tiếng Việt.

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {question}

    Câu trả lời chi tiết:
    """
    
    # Khởi tạo mô hình Gemini-Pro
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Tạo prompt và chuỗi QA
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    """
    Xử lý câu hỏi, tìm kiếm và trả về câu trả lời từ mô hình.
    """
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Vui lòng tải lên và xử lý tài liệu trước khi đặt câu hỏi.")
        return

    try:
        with st.spinner("AI đang suy nghĩ..."):
            # Tìm các tài liệu liên quan trong cơ sở dữ liệu vector
            docs = st.session_state.vector_store.similarity_search(user_question, k=5)
            
            # Lấy chuỗi hỏi đáp
            chain = get_conversational_chain()
            
            # Chạy chuỗi và nhận kết quả
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            
            # Hiển thị câu trả lời
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", response["output_text"]))

            # Hiển thị lại toàn bộ lịch sử chat
            display_chat_history()

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")

def display_chat_history():
    """
    Hiển thị lịch sử cuộc trò chuyện.
    """
    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message)

# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---

def main():
    st.set_page_config(page_title="Chat Với Dữ Liệu Của Bạn", page_icon="📚")

    # Khởi tạo session state nếu chưa có
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- THANH SIDEBAR ---
    with st.sidebar:
        st.title("Thiết lập ⚙️")
        st.markdown("Chào mừng bạn đến với ứng dụng chat với dữ liệu cá nhân, được cung cấp bởi Gemini AI.")

        # Nhập API Key
        try:
            # Cố gắng lấy key từ secrets của Streamlit trước
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("Đã tải API Key từ secrets!", icon="✅")
        except:
            # Nếu không có, yêu cầu người dùng nhập
            api_key = st.text_input("Nhập Google API Key của bạn:", type="password", help="Lấy key tại aistudio.google.com")
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                st.error(f"API Key không hợp lệ: {e}")


        st.subheader("Nguồn tri thức của bạn")
        uploaded_files = st.file_uploader(
            "Tải lên các tệp PDF hoặc DOCX",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )

        if st.button("Xây dựng cơ sở tri thức"):
            if not api_key:
                st.warning("Vui lòng nhập Google API Key trước.")
            elif not uploaded_files:
                st.warning("Vui lòng tải lên ít nhất một tệp tài liệu.")
            else:
                with st.spinner("Đang xử lý tài liệu... Quá trình này có thể mất vài phút."):
                    # 1. Trích xuất văn bản
                    raw_text = get_text_from_files(uploaded_files)
                    
                    if raw_text:
                        # 2. Chia thành các chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Vector hóa và lưu trữ
                        get_vector_store(text_chunks)
                    else:
                        st.error("Không thể trích xuất văn bản từ các tệp đã tải lên.")

    # --- KHUNG CHAT CHÍNH ---
    st.header("Chat Với Dữ Liệu Của Bạn 🤖📚")
    st.write("Tải tài liệu của bạn lên, xây dựng cơ sở tri thức, và bắt đầu hỏi đáp!")
    st.markdown("---")

    # Hiển thị lịch sử chat đã có
    display_chat_history()

    # Ô nhập câu hỏi của người dùng
    user_question = st.chat_input("Đặt câu hỏi về nội dung tài liệu của bạn...")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()

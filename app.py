import os
import uuid
import streamlit as st

# LangChain imports
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.vectorstores import Chroma

# Ollama LLM
try:
    from langchain_ollama import OllamaLLM
except Exception:
    from langchain_community.llms import Ollama as OllamaLLM

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Document Chatbot", layout="wide")

# Custom heading
st.markdown("<h1 style='text-align: center; color: #1E90FF; font-family:Arial;'>💬 Document Chatbot</h1>", unsafe_allow_html=True)

# Instructions section
with st.expander("ℹ️ How to use the chatbot"):
    st.write("""
    1. Upload a PDF or TXT file.  
    2. Type your question in the chat input box.  
    3. Get answers based on document content.  
    4. Click '🧹 Clear Chat' to reset conversation.
    """)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.user-msg {
    background: linear-gradient(135deg, #1E90FF, #00BFFF);
    color: white;
    padding: 10px 15px;
    border-radius: 25px;
    margin: 5px;
    text-align: right;
    float: right;
    clear: both;
    max-width: 70%;
    font-size: 16px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
}
.bot-msg {
    background: linear-gradient(135deg, #333, #555);
    color: #00FF00;
    padding: 10px 15px;
    border-radius: 25px;
    margin: 5px;
    text-align: left;
    float: left;
    clear: both;
    max-width: 70%;
    font-size: 16px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
}
div.stButton > button:first-child {
    background-color: #1E90FF;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 20px;
}
div.stButton > button:first-child:hover {
    background-color: #00BFFF;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Prompt Template ----------------
QA_TEMPLATE = """Answer the question using only the given context.
If the answer is not in the context, say "I don't know from the document."

Context:
{context}

Question: {question}
Answer:"""
qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

# ---------------- Functions ----------------
def load_document(path):
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path).load()

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = os.path.join(".chroma", str(uuid.uuid4())[:8])
    return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=False)

if uploaded_file and st.session_state.qa_chain is None:
    os.makedirs('Uploaded_Files', exist_ok=True)
    save_path = os.path.join("Uploaded_Files", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    docs = load_document(save_path)
    vdb = create_vectorstore(docs)
    retriever = vdb.as_retriever()

    # Default model
    llm = OllamaLLM(model="mistral", temperature=0.2)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )

# ---------------- Chat Flow ----------------
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state.messages.append(("user", user_input))
        st.rerun()

    for i, (role, msg) in enumerate(st.session_state.messages):
        if role == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg}</div>", unsafe_allow_html=True)
            if i == len(st.session_state.messages) - 1:
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_chain.invoke({"query": msg})
                    answer = result.get("result") if isinstance(result, dict) else str(result)
                st.session_state.messages.append(("bot", answer))
                st.rerun()
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg}</div>", unsafe_allow_html=True)

# ---------------- Clear Chat Button ----------------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

import os
import streamlit as st
import tempfile
import re
from collections import OrderedDict

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import GPT4All

# -------------------------
# Helper: Load documents
# -------------------------
def load_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

# -------------------------
# Helper: Parse MCQ options
# -------------------------
def parse_options_from_text(text):
    matches = list(re.finditer(r'([A-Da-d])\s*[)\.\-:]\s*', text))
    if not matches:
        return OrderedDict()
    opts = OrderedDict()
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        option_text = text[start:end].strip()
        option_text = re.sub(r'\s+', ' ', option_text).strip()
        opts[label] = option_text
    return opts

def normalize_text(s):
    s = s or ""
    s = re.sub(r'[^0-9A-Za-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def match_answer_to_options(predicted_answer, options_dict, score_threshold=60):
    from rapidfuzz import process, fuzz

    if not options_dict:
        return None, None, 0

    cleaned_map = []
    labels = []
    for label, opt in options_dict.items():
        cleaned_map.append(normalize_text(opt))
        labels.append(label)

    cleaned_pred = normalize_text(predicted_answer)
    best = process.extractOne(cleaned_pred, cleaned_map, scorer=fuzz.WRatio)
    if not best:
        return None, None, 0

    _, score, idx = best
    best_label = labels[idx]
    best_option_text = options_dict[best_label]

    if score < score_threshold:
        return None, best_option_text, score

    return best_label, best_option_text, score

# -------------------------
# Create Vectorstore using HuggingFace embeddings (CPU)
# -------------------------
def create_vectorstore(split_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ CPU only
    )
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.title("📚 Local RAG Chatbot (HuggingFace CPU + GPT4All, MCQ Support)")

    uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        os.makedirs("uploaded", exist_ok=True)
        file_path = os.path.join("uploaded", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("📑 Loading document...")
        docs = load_document(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split = splitter.split_documents(docs)

        st.write("🧠 Creating knowledge base...")
        vectordb = create_vectorstore(split)
        retriever = vectordb.as_retriever(search_kwargs={"k":3})

        # Local LLM (GPT4All)
        llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", verbose=True)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.session_state.qa_chain = qa_chain
        st.success("✅ Knowledge base ready!")

    if "qa_chain" in st.session_state:
        query = st.text_area("Ask a question (include options if applicable):")
        if st.button("Get Answer"):
            result = st.session_state.qa_chain.run(query)
            predicted = result

            options = parse_options_from_text(query)
            if options:
                label, opt_text, score = match_answer_to_options(predicted, options)
                if label:
                    st.markdown(f"**Answer:** {label}) {opt_text} _(confidence {score:.0f})_")
                else:
                    st.markdown(f"**Predicted answer (no confident option match):** {predicted}")
            else:
                st.markdown(f"**Answer:** {predicted}")

if __name__ == "__main__":
    main()

# 💬 RAG Documentation

## **Project Overview**
The **Document Chatbot** is a web-based application that allows users to upload a PDF or TXT document and ask questions related to its content. It leverages **Retrieval-Augmented Generation (RAG)** with **LangChain** and **Ollama LLM** to provide accurate answers based only on the uploaded document.

---

## **Tech Stack**

- **Python 3.10+**
- **Streamlit** – Frontend web interface
- **LangChain** – Document retrieval and LLM integration
- **Ollama LLM** – Large language model for generating answers
- **HuggingFace Embeddings** – For converting document chunks into embeddings
- **Chroma** – Vector database to store embeddings

---

## **Project Setup**

### **1. Python Environment Setup**
```bash
python -m venv venv
# Activate virtual environment
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install --upgrade pip
```

### **2. Install Dependencies**
```bash
pip install streamlit langchain langchain-ollama chromadb sentence-transformers
```

### **3. Install and Start Ollama**
```bash
# Install Ollama from https://ollama.com
ollama serve
```

### **4. Run Application**
```bash
streamlit run app.py
```

- Access the application at `http://localhost:8501/`

---

## **Project Features**

- Upload PDF/TXT documents
- Ask questions about the document content
- Gradient chat bubbles with shadows
- User (`👤`) and bot (`🤖`) avatars
- Scrollable chat interface
- Clear chat button
- Instructions section for guidance
- Fully responsive and polished UI

---

## **Application Workflow**

1. **File Upload**: Load PDF/TXT using `PyPDFLoader` or `TextLoader`.
2. **Document Chunking**: Split document into chunks for processing.
3. **Vector Embedding**: Convert chunks into embeddings using HuggingFace.
4. **Vector Store**: Store embeddings in Chroma database.
5. **Query Processing**: Retrieve relevant chunks and generate answer using Ollama LLM.
6. **Chat Display**: Display conversation with customized chat bubbles.

---

## **UI Customizations**

- Gradient background for chat bubbles
- Rounded edges with shadows
- Scrollable chat container
- Custom heading: 💬 Document Chatbot
- User and bot avatars
- Clear chat button with hover effect
- Page background color

---

## **Potential Enhancements**

- Multi-file upload support
- Download chat history as `.txt`
- Sidebar for advanced model settings
- Two-column layout: file uploader + chat window
- Advanced prompt engineering for better answer accuracy

---

## **Key Learnings**

- Integration of LLM with document retrieval
- Using embeddings for semantic search
- Building interactive chat applications with Streamlit
- Managing session state and dynamic content
- Polishing UI using CSS for professional look

---

## **References**

1. [LangChain Documentation](https://www.langchain.com/docs/)
2. [Streamlit Documentation](https://docs.streamlit.io/)
3. [Ollama LLM](https://ollama.com/)
4. [HuggingFace Embeddings](https://huggingface.co/sent

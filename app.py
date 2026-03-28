import streamlit as st
import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page config
st.set_page_config(page_title="AI Document Q&A", layout="wide")

# 🎨 Custom Theme (Brown + Cream)
st.markdown("""
<style>
.stApp {
    background-color: #6B4F3A;
    color: #F5F5DC;
}

h1, h2, h3, h4, h5, h6, p, div {
    color: #F5F5DC !important;
}

.stTextInput input {
    background-color: #8B6F4E;
    color: #F5F5DC;
    border-radius: 10px;
}

.stFileUploader {
    background-color: #8B6F4E;
    border-radius: 10px;
    padding: 10px;
}

.chat-user {
    background-color: #A67B5B;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
}

.chat-ai {
    background-color: #8B6F4E;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>📄 AI Document Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a PDF and chat with it</p>", unsafe_allow_html=True)
st.markdown("---")

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")  # stable

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

    st.success("✅ Document ready!")

    # Input
    query = st.text_input("💬 Ask something about your document")

    if query:
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer based only on the context below.

            Context:
            {context}

            Question:
            {query}
            """

            response = model.generate_content(prompt)
            answer = response.text

            st.session_state.chat_history.append((query, answer))

    # Chat display
    st.markdown("### 🧠 Conversation")

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-ai'><b>AI:</b> {a}</div>", unsafe_allow_html=True)
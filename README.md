#  AI Document Q&A Assistant

An AI-powered document analysis and question-answering system built using Retrieval-Augmented Generation (RAG). This application allows users to upload a PDF and ask context-aware questions based on its content.

---

##  Features

-  Upload PDF documents  
-  Semantic search using vector embeddings  
-  Context-aware answers using Google Gemini  
-  Chat-based interface with conversation history  
-  Fast retrieval using FAISS  
-  Interactive UI built with Streamlit  

---

## Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **LLM:** Google Gemini (gemini-2.5-flash)  
- **Embeddings:** HuggingFace (Sentence Transformers)  
- **Vector Store:** FAISS  
- **Framework:** LangChain  

---

## Architecture

PDF → Text Splitting → Embeddings → FAISS → Retrieval → Gemini → Answer
---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/spurthi1459/ai-document-qa.git
cd ai-document-qa

---

### Create virtual environment (optional but recommended)

conda create -n rag_env python=3.11
conda activate rag_env

## Install dependencies

pip install -r requirements.txt

## Set up API key

$env:GOOGLE_API_KEY="your_api_key_here"

## Run application

streamlit run app.py

## Live demo


👩‍💻 Author

Spurthi Pattanashetti

GitHub: https://github.com/spurthi1459
LinkedIn: https://www.linkedin.com/in/spurthi-pattanashetti








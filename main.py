import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

print("Loading document...")

loader = PyPDFLoader("Spurthi-Pattanashetti.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

print("Ready! Ask questions 👇")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    print("Answer:", response.text)
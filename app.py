import os
from dotenv import load_dotenv

# Load ENV
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Gemini + LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Pinecone vectorstore helper
from utils.vectorstore import create_vectorstore
from utils.loader import load_files
from utils.splitter import split_docs

DATA_FOLDER = "data"
INDEX_NAME = "rag-index"

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)


def initialize_rag():
    print("📥 Loading files...")
    docs = load_files(DATA_FOLDER)

    print("✂️ Splitting text...")
    chunks = split_docs(docs)

    print("📌 Uploading into Pinecone...")
    vectorstore = create_vectorstore(
        chunks,
        embeddings,
        INDEX_NAME,
        PINECONE_API_KEY
    )

    return vectorstore


def ask_rag(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    print("\n🤖 Answer:\n")
    print(qa_chain.run(query))


if __name__ == "__main__":
    vs = initialize_rag()

    while True:
        q = input("\n❓ Ask a question (exit to quit): ")
        if q.lower() == "exit":
            break
        ask_rag(q, vs)

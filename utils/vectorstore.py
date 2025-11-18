from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def create_vectorstore(chunks, embeddings, index_name, api_key):

    pc = Pinecone(api_key=api_key)

    # Create index if not exists
    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        print("🆕 Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name
    )

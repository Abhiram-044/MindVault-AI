from app.services.embedding_service import model
from app.services.vector_store import get_user_path
import faiss
import pickle
import numpy as np
import os

def embed_query(query: str):
    vector = model.encode([query])
    return np.array(vector).astype("float32")

def load_user_index(user_id: str):
    index_path, meta_path = get_user_path(user_id)

    print(index_path, meta_path)
    if not os.path.exists(index_path):
        return None, None
    
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

async def retrieve_chunks(
        user_id: str,
        query: str,
        top_k: int = 5,
        distance_threshold: float = 1.5
):
    index, metadata = load_user_index(user_id)

    if index is None:
        return {
            "context": "",
            "sources": []
        }
    
    query_vector = embed_query(query)
    
    distances, indices = index.search(
        query_vector,
        top_k
    )

    retrieved_chunks = []
    sources = []

    for score, idx in zip(distances[0], indices[0]):
        if idx >= len(metadata):
            continue

        if score > distance_threshold:
            continue

        chunk = metadata[idx]

        retrieved_chunks.append(chunk["text"])

        sources.append({
            "chunk_id": chunk["chunk_id"],
            "file_id": chunk["file_id"],
            "score": float(score)
        })

    context = build_context(retrieved_chunks)

    return {
        "context": context,
        "sources": sources
    }

def build_context(chunks, max_chars=6000):
    context = ""
    total_length  = 0

    for i, chunk in enumerate(chunks):
        formatted = f"\n[Document {i+1}]\n{chunk}\n"

        if total_length + len(formatted) > max_chars:
            break

        context += formatted
        total_length += len(formatted)

    return context


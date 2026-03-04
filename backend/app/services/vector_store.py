import os
import faiss
import pickle
import numpy as np

VECTOR_DIR = "app/vector_store"

os.makedirs(VECTOR_DIR, exist_ok=True)

def get_user_path(user_id: str):
    index_path = f"{VECTOR_DIR}/user_{user_id}.index"
    meta_path = f"{VECTOR_DIR}/user_{user_id}_meta.pkl"

    return index_path, meta_path

async def load_or_create_index(user_id: str, dimension: int):
    index_path, meta_path = get_user_path(user_id)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

    else:
        index = faiss.IndexFlatL2(dimension)
        metadata = []

    return index, metadata

async def save_index(user_id, index, metadata):
    index_path, meta_path = get_user_path(user_id)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

async def add_embeddings(user_id, vectors, chunks):
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]

    index, metadata = await load_or_create_index(
        user_id,
        dimension
    )

    index.add(vectors)

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "file_id": chunk["file_id"],
            "text": chunk["text"]
        })

    await save_index(user_id, index, metadata)

    index_path, meta_path = get_user_path(user_id)

    if os.path.exists(index_path) and os.path.exists(meta_path):
        return True
    else:
        return False

def search_similar(user_id, query_vector, k=5):
    index_path, meta_path = get_user_path(user_id)
    
    if not os.path.exists(index_path):
        return []
    
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    query_vector = np.array([query_vector]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []

    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results
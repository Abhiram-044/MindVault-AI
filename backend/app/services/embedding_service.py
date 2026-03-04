from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def generate_embeddings(texts: list[str]):
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings

def embed_chunks(chunk_objects):
    texts = [c["text"] for c in chunk_objects]

    vectors = generate_embeddings(texts)

    return vectors
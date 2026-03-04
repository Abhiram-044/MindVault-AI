from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len
)

def create_chunk_objects(
        text: str,
        file_id: str,
        user_id: str
):
    chunks = text_splitter.split_text(text)

    chunk_objects = []

    for i, chunk in enumerate(chunks):
        chunk_objects.append({
            "chunk_id": f"{file_id}_{i}",
            "file_id": file_id,
            "user_id": user_id,
            "text": chunk
        })

    return chunk_objects
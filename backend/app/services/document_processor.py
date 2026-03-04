from pypdf import PdfReader
import requests
from io import BytesIO
from app.services.chunking_service import create_chunk_objects
from app.services.embedding_service import embed_chunks

def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    pdf_stream = BytesIO(response.content)
    reader = PdfReader(pdf_stream)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def clean_text(text: str) -> str:
    lines = text.split("\n")

    cleaned = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return "\n".join(cleaned)

async def process_document(
    file_path: str,
    file_id: str,
    user_id: str
):
    print(file_path)
    raw_text = extract_text_from_pdf(file_path)

    cleaned_text = clean_text(raw_text)

    chunks = create_chunk_objects(
        cleaned_text,
        file_id,
        user_id
    )

    vectors = embed_chunks(chunks)

    return chunks, vectors
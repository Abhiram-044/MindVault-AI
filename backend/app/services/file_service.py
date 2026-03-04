import os
from fastapi import UploadFile
from uuid import uuid4
from app.storage.supabase_client import supabase
from app.core.config import settings
from app.services.document_processor import process_document
from bson import ObjectId
from app.database.mongodb import get_database
from app.services.vector_store import add_embeddings

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024 #10MB

def validate_file(file: UploadFile):
    filename = file.filename

    if not filename:
        raise ValueError("Invalid filename")
    
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")
    
    return ext

async def upload_to_supbase(user_id: str, file: UploadFile):
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise ValueError("file too large")
    
    unique_name = f"{user_id}/{uuid4()}_{file.filename}"

    supabase.storage.from_(
        settings.SUPABASE_BUCKET
    ).upload(
        unique_name,
        contents,
        {"content-type": file.content_type}
    )

    return unique_name

async def run_document_processing(
        file_id: str,
        file_path: str,
        user_id: str
):
    db = get_database()

    url = get_signed_url(file_path)

    try:
        chunks, vectors = await process_document(url, file_id, user_id)
        embed_exist = await add_embeddings(
            user_id=user_id,
            vectors=vectors,
            chunks=chunks
        )

        await db.files.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$set": {
                    "chunk_content": len(chunks),
                    "embedding_created": embed_exist,
                    "status": "processed"
                }
            }
        )
    except Exception:
        await db.files.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$set": {
                    "embedding_created": False,
                    "status": "failed"
                }
            }
        )

def get_signed_url(path: str):
    res = supabase.storage.from_(
        settings.SUPABASE_BUCKET
    ).create_signed_url(path, 3600)

    return res["signedURL"]
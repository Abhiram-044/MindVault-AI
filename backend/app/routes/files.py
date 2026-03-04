from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from datetime import datetime, timezone
from app.dependencies.auth_dependency import get_current_user
from app.database.mongodb import get_database
from app.services.file_service import validate_file, upload_to_supbase
from app.services.file_service import run_document_processing
from app.models.file import FileModel
from bson import ObjectId

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload")
async def upload_file(
    backround_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    db = get_database()

    try:
        validate_file(file)

        file_path = await upload_to_supbase(
            str(current_user["_id"]),
            file
        )

        file_doc = {
            "user_id": ObjectId(current_user["_id"]),
            "filename": file.filename,
            "file_path": file_path,
            "status": "processing",
            "chunk_content": 0,
            "created_at": datetime.now(timezone.utc)
        }

        file_doc = FileModel(
            user_id = current_user["_id"],
            filename = file.filename,
            file_path = file_path,
            status = "processing",
            chunk_count = 0,
            created_at = datetime.now(timezone.utc)
        )

        result = await db.files.insert_one(
            file_doc.model_dump(by_alias=True, exclude={"id"})
        )

        backround_tasks.add_task(
            run_document_processing,
            str(result.inserted_id),
            file_path,
            current_user["_id"]
        )

        return {
            "message": "File Uploaded Successfully",
            "file_id": str(result.inserted_id)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/{file_id}/status")
async def get_file_status(
    file_id: str,
    current_user=Depends(get_current_user)
):
    db = get_database()

    if not ObjectId.is_valid(file_id):
        raise HTTPException(400, "Invalid file ID format")
    
    file_id_object = ObjectId(file_id)
    user_id = ObjectId(current_user["_id"])
    
    file_data = await db.files.find_one({
        "_id": file_id_object,
        "user_id": user_id
    })

    if not file_data:
        raise HTTPException(404, "File not found")
    
    return {
        "status": file_data["status"]
    }
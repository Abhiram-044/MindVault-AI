from datetime import datetime, timezone
from app.models.base import MongoBaseModel, PyObjectId
from pydantic import Field

class FileModel(MongoBaseModel):
    user_id: PyObjectId
    filename: str
    file_path: str
    status: str = "processing"
    chunk_count: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
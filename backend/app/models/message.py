from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.base import MongoBaseModel, PyObjectId

class Source(BaseModel):
    file_id: PyObjectId
    chunk_id: str
    score: float

class Message(MongoBaseModel):
    session_id: PyObjectId
    user_id: PyObjectId
    role: str
    content: str
    sources: Optional[List[Source]] = []
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
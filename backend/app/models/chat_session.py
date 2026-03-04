from datetime import datetime, timezone
from app.models.base import MongoBaseModel, PyObjectId
from pydantic import Field

class ChatSession(MongoBaseModel):
    user_id: PyObjectId
    title: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
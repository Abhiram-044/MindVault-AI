from datetime import datetime, timezone
from pydantic import EmailStr, Field

from app.models.base import MongoBaseModel

class User(MongoBaseModel):
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
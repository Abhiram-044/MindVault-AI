from pydantic import BaseModel

class ChatMessageRequest(BaseModel):
    session_id: str
    query: str
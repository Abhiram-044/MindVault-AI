from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.dependencies.auth_dependency import get_current_user
from app.services.rag_service import stream_rag_query
from app.schemas.chat_schema import ChatMessageRequest
from bson import ObjectId
from app.database.mongodb import get_database
from app.models.chat_session import ChatSession
from app.models.message import Message
import asyncio

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/session")
async def create_session(user=Depends(get_current_user)):
    db = get_database()

    session = ChatSession(
        user_id=ObjectId(user["_id"]),
        title="New chat"
    )

    result = await db.chat_sessions.insert_one(
        session.model_dump(by_alias=True, exclude={"id"})
    )
    print(result)

    return {
        "_id": str(result.inserted_id),
        "user_id": user["_id"],
        "title": "New chat",
        "created_at": session.created_at
    }

@router.post("/message/stream")
async def stream_message(
    data: ChatMessageRequest,
    user=Depends(get_current_user)
):
    await asyncio.sleep(5)
    db = get_database()

    sessions = await db.chat_sessions.find_one({
        "_id": ObjectId(data.session_id),
        "user_id": ObjectId(user["_id"])
    })

    if not sessions:
        raise HTTPException(404, "Session not found.")
    
    user_msg = Message(
        session_id=ObjectId(data.session_id),
        user_id=ObjectId(user["_id"]),
        role="user",
        content=data.query
    )

    await db.messages.insert_one(
        user_msg.model_dump(
            by_alias=True,
            exclude={"id"}
        )
    )

    recent_messages = await get_recent_messages(
        db,
        data.session_id
    )
    history = await format_conversation(recent_messages)

    async def event_generator():
        final_answer = ""
        sources = []

        async for chunk in stream_rag_query(
            user_id=str(user["_id"]),
            query=data.query,
            history=history
        ):
            final_answer = chunk.get("full_answer", final_answer)
            sources = chunk.get("sources", [])

            token = chunk["token"]

            yield f"data: {token}\n\n"

        assitant_message = Message(
            session_id=ObjectId(data.session_id),
            user_id=ObjectId(user["_id"]),
            role="assistant",
            content=final_answer,
            sources=sources
        )

        await db.messages.insert_one(
            assitant_message.model_dump(
                by_alias=True,
                exclude={"id"}
            )
        )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@router.get("/sessions")
async def list_sessions(user=Depends(get_current_user)):
    db = get_database()

    sessions = await db.chat_sessions.find(
        {"user_id": ObjectId(user["_id"])}
    ).sort("created_at", -1).to_list(100)

    for s in sessions:
        s["_id"] = str(s["_id"])
        s["user_id"] = str(s["user_id"])

    return sessions

@router.get("/{session_id}")
async def get_messages(
    session_id: str,
    user=Depends(get_current_user)
):
    db = get_database()

    session = await db.chat_sessions.find_one({
        "_id": ObjectId(session_id),
        "user_id": ObjectId(user["_id"])
    })

    if not session:
        raise HTTPException(404, "Session not found")

    messages = await db.messages.find({
        "session_id": ObjectId(session_id)
    }).sort("created_at", 1).to_list(500)

    exclude_keys = {"_id", "user_id", "session_id", "sources", "created_at"}

    return [
        {k: v for k, v in m.items() if k not in exclude_keys}
        for m in messages
    ]

@router.delete("/session/{session_id}")
async def delete_messages(
    session_id: str,
    user=Depends(get_current_user)
):
    db = get_database()

    session = await db.chat_sessions.find_one({
        "_id": ObjectId(session_id),
        "user_id": ObjectId(user["_id"]) 
    })

    if not session:
        raise HTTPException(404, "session not found")
    
    await db.chat_sessions.delete_one({
        "_id": ObjectId(session_id)
    })

    await db.messages.delete_many({
        "session_id": ObjectId(session_id)
    })

    return {"messages": "Deleted"}

async def get_recent_messages(db, session_id, limit=6):
    messages = await db.messages.find(
        {"session_id": ObjectId(session_id)}
    ).sort("created_at", -1).to_list(limit)

    return list(reversed(messages))


async def format_conversation(messages):
    history = ""

    for msg in messages:
        role = msg["role"].capitalize()
        history += f"{role}: {msg["content"]}\n"

    return history
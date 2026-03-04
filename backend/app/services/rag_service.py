from app.services.retrieval_service import retrieve_chunks
from app.services.llm_service import stream_generate_response

async def stream_rag_query(user_id: str, query: str, history: str):
    retrieval_result = await retrieve_chunks(
        user_id=user_id,
        query=query
    )

    if not retrieval_result["context"]:
        yield {
            "token": "I couldn't find relevant knowledge.",
            "done": True,
            "sources": []
        }
        return
    
    async for token, full_answer in stream_generate_response(
        query,
        retrieval_result["context"],
        history
    ):
        yield {
            "token": token,
            "done": False,
            "sources": retrieval_result["sources"],
            "full_answer": full_answer
        }
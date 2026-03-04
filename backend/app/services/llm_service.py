from groq import Groq
from app.core.config import settings

client = Groq(api_key=settings.GROQ_API_KEY)


async def stream_generate_response(query: str, context: str, history: str):
    SYSTEM_PROMPT = f"""
You are an MindVault AI assistant answering ONLY from provided context.

Rules:
- Answer ONLY using provided context
- Use conversation history for continuity
- If answer not found, say you don't know
- Do NOT fabricate information
- Be concise and accurate
- Do not hallucinate

Coversation History:
-------------------
{history}
"""

    USER_PROMPT = f"""
Context:
--------
{context}

Question:
{query}
"""

    stream = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.2,
        stream=True
    )

    full_answer = ""

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_answer += token
        yield token, full_answer
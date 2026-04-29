from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI

import os
import json
import math

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

BASE_DIR = os.path.dirname(__file__)
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store.json")

conversation_histories = {}
MAX_HISTORY_MESSAGES = 12


class ChatRequest(BaseModel):
    message: str
    session_id: str


def load_vector_store():
    with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


def retrieve_relevant_chunks(query: str, top_k=5):
    vector_store = load_vector_store()
    query_emb = get_embedding(query)

    scored = []

    for item in vector_store:
        score = cosine_similarity(query_emb, item["embedding"])

        scored.append({
            "score": score,
            "text": item["text"],
            "filename": item["filename"]
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def explicitly_about_yifan(query: str):
    q = query.lower()

    keywords = [
        "yifan", "ren yifan",
        "you", "your", "yourself",
        "任一凡", "一凡", "你", "你的", "自己"
    ]

    return any(k in q for k in keywords)


def build_retrieval_query(query: str, history: list) -> str:
    """For short follow-up queries, prepend the last exchange so retrieval has context."""
    if len(query.split()) > 8 or not history:
        return query

    last_user = ""
    last_assistant = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        elif msg["role"] == "user" and not last_user:
            last_user = msg["content"]
        if last_user and last_assistant:
            break

    if last_user:
        return f"{last_user} {last_assistant} {query}"
    return query


def generate_response(query: str, session_id: str):
    history = conversation_histories.get(session_id, [])

    retrieval_query = build_retrieval_query(query, history)

    chunks = retrieve_relevant_chunks(retrieval_query, top_k=5)

    if not chunks:
        return "I don't have enough information to answer that."

    max_score = chunks[0]["score"]

    if max_score < 0.3:
        if explicitly_about_yifan(retrieval_query):
            return "I don't have enough information about that yet. 我的资料库里暂时没有足够信息回答这个问题。"
        else:
            return "I can only answer questions related to Yifan. 我只能回答和任一凡相关的问题。"

    context = ""
    for c in chunks:
        context += "\n\n" + c["text"]

    prompt = f"""
You are Yifan's personal AI agent.

IMPORTANT:
- "you", "your", "yourself" refer to Yifan.
- "任一凡" and "一凡" also refer to Yifan.
- The user may ask in English or Chinese only.
- If the user uses any language other than English or Chinese, say:
  "Please ask questions in English or Chinese only."

You must answer as if you are Yifan speaking directly.

Rules:
- Use first person: "I", "my", "me"
- Answer only based on the provided context and recent conversation history
- Be natural and conversational
- Do not mention the context or file names
- If unsure, say you don't know
- Do not make up information

Relevant context about Yifan:
{context}

Current user question:
{query}

Answer:
"""

    messages = [
        {
            "role": "system",
            "content": "You are Yifan. Answer in first person using only the provided context and recent conversation history."
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4
    )

    answer = response.choices[0].message.content

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    conversation_histories[session_id] = history[-MAX_HISTORY_MESSAGES:]

    return answer


@app.post("/chat")
def chat(req: ChatRequest):
    reply = generate_response(req.message, req.session_id)
    return {"reply": reply}
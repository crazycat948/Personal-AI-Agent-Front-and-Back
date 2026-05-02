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

# Only stores relevant Q&A pairs per session
relevant_contexts: dict[str, list[dict]] = {}
MAX_CONTEXT = 5


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
    scored = [
        {
            "score": cosine_similarity(query_emb, item["embedding"]),
            "text": item["text"],
            "filename": item["filename"]
        }
        for item in vector_store
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def classify_query(query: str) -> str:
    """
    Classifies the query into one of three categories:
    - 'relevant'  : directly about Yifan
    - 'followup'  : a continuation/clarification of the previous topic
    - 'irrelevant': not about Yifan, or asking about another specific person/entity
    """
    system_prompt = """\
You are a classifier for a personal AI chatbot about Yifan (任一凡).

Your only job: classify the question into one of three categories.
Output exactly one word: relevant / followup / irrelevant

───────────────────────────────
RELEVANT
───────────────────────────────
The question is asking about Yifan himself — his life, background, personality,
hobbies, preferences, goals, experiences, or his opinion/feeling about something.

✓ relevant examples:
  "what's your hobby?"
  "where did you study?"
  "do you like NewJeans?"          ← asking Yifan's preference
  "what do you think of Lebron?"   ← asking Yifan's opinion
  "你喜欢什么音乐"
  "你的梦想是什么"
  "你玩什么游戏"

───────────────────────────────
FOLLOWUP
───────────────────────────────
A short continuation of the previous topic. No new named subject is introduced.
These are vague phrases that only make sense in context.

✓ followup examples:
  "why?"  "第二呢"  "tell me more"  "and then?"  "为什么"  "the second one?"  "how about that?"

───────────────────────────────
IRRELEVANT
───────────────────────────────
The question is asking for factual information about someone or something that is NOT Yifan,
or asking for general world knowledge. This includes:
- "who is [person]?" where the person is not Yifan
- "what is [thing]?"
- sports results, news, trivia, definitions, general facts

✓ irrelevant examples:
  "who is Lebron James?"      ← asking who Lebron is, not Yifan's opinion
  "who is Taylor Swift?"
  "what is GPT-4?"
  "今天天气怎么样"
  "who won the NBA finals?"
  "写一首诗"
  "help me with my homework"

KEY RULE — the "who is X?" pattern:
  "who is Lebron James?"   → irrelevant  (asking about Lebron, not Yifan)
  "do you like Lebron?"    → relevant    (asking Yifan's feeling)
  "what do you think of X?"→ relevant    (asking Yifan's opinion)

Return ONLY the single word: relevant, followup, or irrelevant"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0,
        max_tokens=10
    )
    result = response.choices[0].message.content.strip().lower()
    if result.startswith("relevant"):
        return "relevant"
    if result.startswith("followup") or result.startswith("follow"):
        return "followup"
    return "irrelevant"


def answer_question(query: str, chunks: list, prior_qa: dict | None) -> str:
    context = "\n\n".join(c["text"] for c in chunks)

    prior_section = ""
    if prior_qa:
        prior_section = f"""Previous relevant exchange:
Q: {prior_qa['question']}
A: {prior_qa['answer']}

"""

    prompt = f"""{prior_section}Relevant context about Yifan:
{context}

Current question: {query}

Answer as Yifan in first person. Be natural and conversational. \
Only use information from the context above. If the answer isn't there, say you're not sure."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are Yifan (任一凡). Answer in first person using only the provided context."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content


def generate_response(query: str, session_id: str) -> str:
    context_list = relevant_contexts.get(session_id, [])
    has_context = len(context_list) > 0

    category = classify_query(query)

    # followup with no prior relevant context → treat as irrelevant
    if category == "irrelevant" or (category == "followup" and not has_context):
        return "I can only answer questions related to Yifan. 我只能回答和任一凡相关的问题。"

    if category == "relevant":
        chunks = retrieve_relevant_chunks(query, top_k=5)
        if not chunks or chunks[0]["score"] < 0.3:
            return "I don't have enough information about that yet. 我的资料库里暂时没有足够信息回答这个问题。"
        answer = answer_question(query, chunks, prior_qa=None)
        updated = context_list + [{"question": query, "answer": answer}]
        relevant_contexts[session_id] = updated[-MAX_CONTEXT:]
        return answer

    # followup with context
    chunks = retrieve_relevant_chunks(query, top_k=3)
    prior_qa = context_list[-1]
    return answer_question(query, chunks, prior_qa=prior_qa)


@app.post("/chat")
def chat(req: ChatRequest):
    reply = generate_response(req.message, req.session_id)
    return {"reply": reply}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

# 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- 请求格式 --------
class ChatRequest(BaseModel):
    query: str


# -------- 读取知识库 --------
def load_data():

    data_text = ""

    folder = "backend/data"

    if not os.path.exists(folder):
        return "Knowledge folder not found."

    for file in os.listdir(folder):

        if file.endswith(".md"):

            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8") as f:
                data_text += f"\n\n--- {file} ---\n"
                data_text += f.read()

    return data_text


# -------- 模拟 AI 回复 --------
def generate_response(query: str):

    knowledge = load_data()

    q = query.lower()

    # 简单关键词匹配
    if "philosophy" in q:
        return knowledge[:800]

    if "hobby" in q or "music" in q or "anime" in q:
        return knowledge[:800]

    if "love" in q:
        return knowledge[:800]

    if "who are you" in q:
        return "I'm Yifan's personal AI assistant. I can talk about his projects, hobbies, philosophy, and ideas."

    # 默认回答
    return (
        "I know many things about Yifan from his knowledge base. "
        "Ask me about his hobbies, philosophy, projects, or thoughts."
    )

# -------- API 路由 --------
@app.post("/chat")
def chat(req: ChatRequest):

    reply = generate_response(req.query)

    return {"reply": reply}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

# ---- 模拟 AI 回复函数 ----
def generate_fake_response(query: str) -> str:
    q = query.lower()

    if "project" in q:
        return "Yifan has built several projects including SkillVault, a full-stack personal skill tracking platform."

    elif "music" in q:
        return "Yifan enjoys rap, indie music, and exploring new underground artists."

    elif "game" in q:
        return "Yifan likes strategy games, RPGs, and occasionally competitive online games."

    elif "literature" in q or "book" in q:
        return "Yifan is interested in literature, especially works that explore philosophy and human nature."

    elif "who are you" in q:
        return "I'm Yifan's personal AI assistant, designed to talk about his projects, interests, and ideas."

    else:
        return "That's an interesting question! Tell me more about what you'd like to know about Yifan."

# ---- API 路由 ----
@app.post("/chat")
def chat(req: ChatRequest):
    reply = generate_fake_response(req.query)
    return {"reply": reply}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import OpenAI

import os

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str


# Load data from md files
def load_data():

    data_text = ""

    BASE_DIR = os.path.dirname(__file__)
    DATA_FOLDER = os.path.join(BASE_DIR, "data")

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".md"):

            path = os.path.join(DATA_FOLDER, file)

            with open(path, "r", encoding="utf-8") as f:
                data_text += f"\n\n--- {file} ---\n"
                data_text += f.read()

    return data_text


# Generate AI response
def generate_response(query: str):

    data = load_data()

    prompt = f"""
You are Yifan's personal AI assistant.

Use the following information about Yifan to answer the question.

Information about Yifan:

{data}

User question:
{query}

Answer naturally and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Yifan's personal AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    reply = generate_response(request.message)

    return {"reply": reply}

🚀 Personal AI Agent (RAG-based)

A full-stack personal AI agent that answers questions about me using a Retrieval-Augmented Generation (RAG) pipeline.

👉 Live Demo:
https://ai-agent-frontend-s3aj.onrender.com/

🧠 Overview

This project implements a single-agent AI system that retrieves relevant information from a custom knowledge base and generates responses using an LLM.

Unlike many tutorials, this project is built from scratch without using frameworks like LangChain, to better understand the underlying mechanics of RAG systems.

⚙️ Tech Stack
Backend
Python
FastAPI
OpenAI API (LLM + embeddings)
Custom RAG pipeline (no LangChain)
Frontend
HTML / CSS / JavaScript
Vanilla JS chat interface
Deployment
Render (backend + frontend)
🔍 Features
💬 Chat-based interface
🧠 RAG (Retrieval-Augmented Generation)
📚 Custom knowledge base (personal data)
🔎 Embedding-based similarity search
🧾 Multi-turn conversation support
🌐 Fully deployed (public access)
🧱 System Architecture
User Input (Frontend)
        ↓
Fetch API
        ↓
FastAPI Backend
        ↓
Embedding + Retrieval
        ↓
Context Injection
        ↓
LLM (OpenAI)
        ↓
Response → Frontend
🧠 How It Works
1. Knowledge Base
Stored as text files (about me, projects, etc.)
Chunked into smaller pieces
Converted into embeddings
Saved in a local vector store (vector_store.json)
2. Retrieval (RAG)

When a user asks a question:

The query is embedded
Similar chunks are retrieved using cosine similarity
Relevant context is injected into the prompt
3. Prompting

The model is guided with rules such as:

Only answer questions related to me
Avoid unrelated general knowledge
Use retrieved context when possible
4. Memory (Multi-turn)

A lightweight session-based memory system is implemented:

conversation_histories.get(session_id, [])
Each session stores past messages
Previous context is appended to each request
Enables multi-turn conversations without external frameworks
🚀 Deployment

Both frontend and backend are deployed using Render:

Backend → Web Service (FastAPI)
Frontend → Static Site
📦 Setup (Local Development)
1. Clone repo
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Backend
cd backend
pip install -r requirements.txt

Create .env:

OPENAI_API_KEY=your_api_key

Run:

uvicorn main:app --reload
3. Frontend

Open:

frontend/index.html
🔄 Updating Knowledge Base

After modifying data files:

cd backend
python build_index.py

Then commit:

git add backend/vector_store.json
git commit -m "update vector store"
git push
⚠️ Known Issues
The agent may sometimes ignore guardrails after long conversations
Example: answering unrelated questions like "Who is LeBron James?"
Attempts made:
Query rewriting
Prompt engineering
LLM-based classification

Still exploring better solutions for alignment and control.

something it says error to connect with endfront just fresh the page and try it again.

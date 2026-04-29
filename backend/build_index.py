import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "vector_store.json")

EMBEDDING_MODEL = "text-embedding-3-small"


def read_md_files():
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def build_index():
    documents = read_md_files()
    vector_store = []

    print(f"Found {len(documents)} markdown files.")

    for doc in documents:
        filename = doc["filename"]
        chunks = chunk_text(doc["text"])

        print(f"Processing {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            vector_store.append({
                "filename": filename,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vector_store, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(vector_store)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_index()
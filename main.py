import os
import uuid
import sqlite3
import base64
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Surf AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("OPENAI KEY EXISTS:", bool(api_key))
print("WEB SEARCH KEY EXISTS:", bool(tavily_key))

client = OpenAI(
    api_key=api_key,
    timeout=60
)

DB_PATH = "/data/surf_memory.db" if os.path.exists("/data") else "surf_memory.db"

db = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)
cursor = db.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    user_id TEXT,
    role TEXT,
    content TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = None
    user_id: str | None = None

class RenameRequest(BaseModel):
    title: str

def search_the_web(query: str) -> str:
    if not tavily_key:
        return "[System Notice: Live web search is currently unavailable because TAVILY_API_KEY is missing on Render.]"
    try:
        url = "https://tavily.com"
        payload = {
            "api_key": tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            context_string = "\n--- LIVE WEB SEARCH RESULTS ---\n"
            for i, res in enumerate(results, 1):
                context_string += f"Source Mapping Index: [{i}]\nSource Name: {res.get('title')}\nURL: {res.get('url')}\nContent Summary: {res.get('content')}\n\n"
            return context_string
        return "[System Notice: Web search API failed to respond properly.]"
    except Exception as e:
        return f"[System Notice: Web search exception occurred: {str(e)}]"

@app.get("/")
def home():
    return {"status": "Surf AI is running"}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("INCOMING INTERFACE ACTION:", body)
        conversation_id = (
            body.get("conversation_id")
            or body.get("session_id")
            or body.get("chat_id")
            or str(uuid.uuid4())
        )
        message = body.get("message") or body.get("content") or ""
        user_id = body.get("user_id") or "guest"
        file_data = body.get("file")
        location_data = body.get("location")

        if not message and not file_data:
            return {
                "error": "Empty message and no files provided",
                "conversation_id": conversation_id
            }

        cursor.execute(
            "INSERT OR IGNORE INTO conversations (conversation_id, title) VALUES (?, ?)",
            (conversation_id, message[:30] if message else "New Chat")
        )
        db.commit()

        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
            (conversation_id,)
        )
        old_messages = cursor.fetchall()[::-1]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Surf AI, a friendly and helpful assistant created by an aerospace engineering and AI/ML student at Université Paris-Saclay. "
                    "If a user asks who made you, who created you, or about your developer, always state clearly that you were built by this student at Université Paris-Saclay. "
                    "CRITICAL SECURITY PREROGATIVE: This project places an absolute, ultra-high emphasis on user privacy. "
                    "If asked about logging, tracking, or watch lists, reassure them warmly and firmly that their records are private and accessible only to their runtime index. "
                    "LIVE SEARCH ENGINE CONNECTIONS: You have instant access to Google/Yahoo live web resources via embedded text blocks below labeled '--- LIVE WEB SEARCH RESULTS ---'. "
                    "Always prioritize this up-to-date information to resolve timelines or topical events correctly. "
                    "COMPACT INLINE CITATIONS ONLY: When incorporating details from search records, cite them inline exactly using markdown links where the "
                    "visible display text is a numbered reference value matching the source index, formatted precisely like [1](URL) or [2](URL). "
                    "DO NOT print out a general bibliography, data index list, or URL breakdown summary section at the bottom of the chat bubble. Write inline badges organically as you draft."
                )
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        current_content = []
        db_log_content = message 

        if message:
            search_data = search_the_web(message)
            current_content.append({"type": "text", "text": search_data})

        if location_data:
            lat = location_data.get("latitude")
            lon = location_data.get("longitude")
            current_content.append({
                "type": "text", 
                "text": f"\n[System Log: User granted location access. Approximate Coordinates: Lat {lat}, Lon {lon}]\n"
            })

        if message:
            current_content.append({"type": "text", "text": f"User Prompt: {message}"})

        if file_data:
            file_name = file_data.get("name", "uploaded_file")
            file_type = file_data.get("type", "")
            base64_str = file_data.get("base64", "")
            if "image" in file_type:
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{file_type};base64,{base64_str}"}
                })
                db_log_content += f" [Attached Image: {file_name}]"
            else:
                try:
                    decoded_text = base64.b64decode(base64_str).decode("utf-8")
                    file_context = f"\n\n[Contents of Document '{file_name}']:\n{decoded_text}"
                    current_content.append({"type": "text", "text": file_context})
                    db_log_content += f" [Attached Text Document: {file_name}]"
                except Exception:
                    current_content.append({"type": "text", "text": f"\n\n[Attached Unreadable File: {file_name}]"})

        cursor.execute(
            "INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, "user", db_log_content)
        )
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
            (conversation_id,)
        )
        db.commit()

        messages.append({"role": "user", "content": current_content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

        cursor.execute(
            "INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, "assistant", reply)
        )
        db.commit()

        return {
            "reply": reply,
            "conversation_id": conversation_id
        }
    except Exception as e:
        print("ERROR LOGGED:", str(e))
        return {"error": str(e)}

@app.get("/history")
def get_all_conversations():
    try:
        cursor.execute("SELECT conversation_id, title FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [{"conversation_id": row[0], "title": row[1]} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/history/{conversation_id}/rename")
def rename_conversation(conversation_id: str, request: RenameRequest):
    try:
        cursor.execute("UPDATE conversations SET title = ? WHERE conversation_id = ?", (request.title, conversation_id))
        db.commit()
        return {"status": "success", "message": f"Conversation renamed to {request.title}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{conversation_id}")
def get_history(conversation_id):
    try:
        cursor.execute("SELECT title FROM conversations WHERE conversation_id=?", (conversation_id,))
        title_row = cursor.fetchone()
        chat_title = title_row[0] if title_row else "Untitled Chat"
        cursor.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC", (conversation_id,))
        rows = cursor.fetchall()
        return {
            "conversation_id": conversation_id,
            "title": chat_title,
            "messages": [{"role": row[0], "content": row[1]} for row in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{conversation_id}")
def delete_history(conversation_id):
    try:
        cursor.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
        db.commit()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

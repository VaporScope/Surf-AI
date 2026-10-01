import os
import uuid
import sqlite3
import base64

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Surf AI API")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change later to your Base44 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OPENAI
# =========================
api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI KEY EXISTS:", bool(api_key))

client = OpenAI(
    api_key=api_key,
    timeout=60
)

# =========================
# DATABASE (Persistent Path for Render)
# =========================
DB_PATH = "/data/surf_memory.db" if os.path.exists("/data") else "surf_memory.db"

db = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)
cursor = db.cursor()

# 1. Table for message history strings
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    user_id TEXT,
    role TEXT,
    content TEXT
)
""")

# 2. NEW: Table to manage custom chat session names/titles
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

# =========================
# REQUEST MODELS
# =========================
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = None
    user_id: str | None = None

class RenameRequest(BaseModel):
    title: str

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"status": "Surf AI is running"}

# =========================
# CHAT
# =========================
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("INCOMING PAYLOAD:", body)

        conversation_id = (
            body.get("conversation_id")
            or body.get("session_id")
            or body.get("chat_id")
            or str(uuid.uuid4())
        )

        message = body.get("message") or body.get("content") or ""
        user_id = body.get("user_id") or "guest"
        file_data = body.get("file")

        if not message and not file_data:
            return {
                "error": "Empty message and no files provided",
                "conversation_id": conversation_id
            }

        # Automatically register conversation in titles database if it's new
        cursor.execute(
            "INSERT OR IGNORE INTO conversations (conversation_id, title) VALUES (?, ?)",
            (conversation_id, message[:30] if message else "New Chat")
        )
        db.commit()

        # Load memory history frames
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
            (conversation_id,)
        )
        old_messages = cursor.fetchall()[::-1]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Surf AI, a friendly and helpful assistant. "
                    "You were created by an aerospace engineering and AI/ML student at Université Paris-Saclay. "
                    "If a user asks who made you, who created you, or about your developer, always state clearly that you were built by this student at Université Paris-Saclay. "
                    "You can inspect uploaded images or read text contents from uploaded documents. "
                    "Remember details from earlier in the conversation history to answer follow-up prompts smoothly."
                )
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        current_content = []
        db_log_content = message 

        if message:
            current_content.append({"type": "text", "text": message})

        if file_data:
            file_name = file_data.get("name", "uploaded_file")
            file_type = file_data.get("type", "")
            base64_str = file_data.get("base64", "")

            if "image" in file_type:
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{file_type};base64,{base64_str}"}
                })
                db_log_content += f" [Sent Image: {file_name}]"
            else:
                try:
                    decoded_text = base64.b64decode(base64_str).decode("utf-8")
                    file_context = f"\n\n[Attached File Context '{file_name}']:\n{decoded_text}"
                    current_content.append({"type": "text", "text": file_context})
                    db_log_content += f" [Attached Document: {file_name}]"
                except Exception:
                    current_content.append({"type": "text", "text": f"\n\n[Attached unreadable file placeholder: {file_name}]"})

        # Record user entry
        cursor.execute(
            "INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, "user", db_log_content)
        )
        db.commit()

        messages.append({"role": "user", "content": current_content})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices[0].message.content

        # Record AI generation
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
        print("ERROR:", str(e))
        return {"error": str(e)}

# =========================
# NEW: RENAME CONVERSATION ENDPOINT
# =========================
@app.put("/history/{conversation_id}/rename")
def rename_conversation(conversation_id: str, request: RenameRequest):
    try:
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (request.title, conversation_id)
        )
        db.commit()
        return {"status": "success", "message": f"Conversation renamed to {request.title}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# GET CHAT HISTORY & TITLE
# =========================
@app.get("/history/{conversation_id}")
def get_history(conversation_id):
    # Fetch title
    cursor.execute("SELECT title FROM conversations WHERE conversation_id=?", (conversation_id,))
    title_row = cursor.fetchone()
    chat_title = title_row[0] if title_row else "Untitled Chat"

    # Fetch rows
    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conversation_id,)
    )
    rows = cursor.fetchall()

    return {
        "conversation_id": conversation_id,
        "title": chat_title,
        "messages": [
            {"role": role, "content": content}
            for role, content in rows
        ]
    }

# =========================
# DELETE CHAT
# =========================
@app.delete("/history/{conversation_id}")
def delete_history(conversation_id):
    cursor.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
    db.commit()
    return {"status": "deleted"}

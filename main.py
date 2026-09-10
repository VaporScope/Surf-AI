import os
import uuid
import sqlite3

from fastapi import FastAPI, Request
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
# Render wipes local files unless saved in a persistent disk directory like /data/
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
db.commit()


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = None
    user_id: str | None = None


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "status": "Surf AI is running"
    }


# =========================
# CHAT
# =========================

@app.post("/chat")
async def chat(request: Request):

    try:
        # Read whatever Base44 sends
        body = await request.json()
        print("INCOMING PAYLOAD:", body)

        # 1. Base44 compatibility fallback check
        # Some frontend engines use session_id or chat_id instead of conversation_id
        conversation_id = (
            body.get("conversation_id")
            or body.get("session_id")
            or body.get("chat_id")
            or str(uuid.uuid4())
        )

        message = (
            body.get("message")
            or body.get("content")
            or ""
        )

        user_id = (
            body.get("user_id")
            or "guest"
        )

        if not message:
            return {
                "error": "Empty message",
                "conversation_id": conversation_id
            }

        # 2. Load previous memory (Limit to last 20 messages to prevent token limits)
        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id=?
            ORDER BY id DESC
            LIMIT 20
            """,
            (conversation_id,)
        )
        
        # We read DESC to get the latest messages, then reverse it to read chronologically
        old_messages = cursor.fetchall()[::-1]

        # 3. Construct System Prompt Context
        messages = [
            {
                "role": "system",
                "content": "You are Surf AI, a friendly helpful assistant. Remember details from earlier in the conversation history to answer follow-up prompts smoothly."
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        # 4. Save incoming user message to SQLite memory
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, user_id, "user", message)
        )
        db.commit()

        # Add current user message to API payload
        messages.append({"role": "user", "content": message})

        # 5. Send complete thread to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = response.choices[0].message.content

        # 6. Save AI reply to SQLite memory
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, user_id, "assistant", reply)
        )
        db.commit()

        # Base44 requires a consistent format. 
        # Always return the conversation_id back so the frontend can reuse it!
        return {
            "reply": reply,
            "conversation_id": conversation_id
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }


# =========================
# GET CHAT HISTORY
# =========================

@app.get("/history/{conversation_id}")
def get_history(conversation_id):
    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id=?
        ORDER BY id ASC
        """,
        (conversation_id,)
    )
    rows = cursor.fetchall()

    return {
        "conversation_id": conversation_id,
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
    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id=?
        """,
        (conversation_id,)
    )
    db.commit()

    return {"status": "deleted"}

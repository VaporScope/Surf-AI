import os
import sqlite3

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI(title="Surf AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change this to your Base44 domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI
# ----------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60.0
)

# ----------------------------
# Database
# ----------------------------

conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
)
""")

conn.commit()

# ----------------------------
# Models
# ----------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def home():
    return {"status": "Surf AI is running"}

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        # Load previous conversation
        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id=?
            ORDER BY id
            """,
            (request.conversation_id,)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Surf AI, a friendly helpful assistant."
                )
            }
        ]

        for role, content in cursor.fetchall():
            messages.append({
                "role": role,
                "content": content
            })

        # Save user's new message
        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (
                request.conversation_id,
                "user",
                request.message
            )
        )

        conn.commit()

        messages.append({
            "role": "user",
            "content": request.message
        })

        # Ask OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = response.choices[0].message.content

        # Save assistant reply
        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (
                request.conversation_id,
                "assistant",
                reply
            )
        )

        conn.commit()

        return {
            "reply": reply
        }

    except Exception as e:
        print(e)
        return {
            "error": str(e)
        }

@app.post("/clear/{conversation_id}")
def clear_chat(conversation_id: str):

    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id=?
        """,
        (conversation_id,)
    )

    conn.commit()

    return {
        "status": "Conversation cleared"
    }

@app.get("/history/{conversation_id}")
def history(conversation_id: str):

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id=?
        ORDER BY id
        """,
        (conversation_id,)
    )

    rows = cursor.fetchall()

    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]
    }




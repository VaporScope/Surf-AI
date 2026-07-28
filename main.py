import os
import sqlite3
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI(title="Surf AI API")


# ----------------------------
# CORS
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://asksurf.base44.app",
        "https://app.base44.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# OpenAI
# ----------------------------

api_key = os.getenv("OPENAI_API_KEY")

print("OPENAI KEY EXISTS:", bool(api_key))

client = OpenAI(
    api_key=api_key,
    timeout=60
)


# ----------------------------
# Database
# ----------------------------

conn = sqlite3.connect(
    "chat_history.db",
    check_same_thread=False
)

cursor = conn.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()


# ----------------------------
# Request Model
# ----------------------------

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = None
    user_id: str | None = None


# ----------------------------
# Home
# ----------------------------

@app.get("/")
def home():
    return {
        "status": "Surf AI is running"
    }


# ----------------------------
# Chat
# ----------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        print("REQUEST RECEIVED:")
        print(request)


        # Create conversation if missing
        conversation_id = request.conversation_id

        if not conversation_id:
            conversation_id = str(uuid.uuid4())


        # Protect against empty messages
        if not request.message:
            return {
                "error": "No message received"
            }


        # Load history

        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id=?
            ORDER BY id ASC
            """,
            (conversation_id,)
        )


        history = cursor.fetchall()


        messages = [
            {
                "role": "system",
                "content": 
                "You are Surf AI, a friendly helpful assistant."
            }
        ]


        for role, content in history:
            messages.append(
                {
                    "role": role,
                    "content": content
                }
            )


        # Save user message

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id,user_id,role,content)
            VALUES (?,?,?,?)
            """,
            (
                conversation_id,
                request.user_id,
                "user",
                request.message
            )
        )

        conn.commit()


        messages.append(
            {
                "role": "user",
                "content": request.message
            }
        )


        # OpenAI

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )


        reply = response.choices[0].message.content


        # Save AI response

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id,user_id,role,content)
            VALUES (?,?,?,?)
            """,
            (
                conversation_id,
                request.user_id,
                "assistant",
                reply
            )
        )

        conn.commit()


        return {
            "reply": reply,
            "conversation_id": conversation_id
        }


    except Exception as e:

        print("ERROR:")
        print(str(e))

        return {
            "error": str(e)
        }



# ----------------------------
# Get History
# ----------------------------

@app.get("/history/{conversation_id}")
def get_history(conversation_id):

    cursor.execute(
        """
        SELECT role,content
        FROM messages
        WHERE conversation_id=?
        ORDER BY id
        """,
        (conversation_id,)
    )


    data = cursor.fetchall()


    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "role": role,
                "content": content
            }
            for role,content in data
        ]
    }



# ----------------------------
# Clear Chat
# ----------------------------

@app.delete("/history/{conversation_id}")
def clear_history(conversation_id):

    cursor.execute(
        """
        DELETE FROM messages
        WHERE conversation_id=?
        """,
        (conversation_id,)
    )

    conn.commit()


    return {
        "status":"deleted"
    }

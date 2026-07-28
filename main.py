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
# DATABASE
# =========================

db = sqlite3.connect(
    "surf_memory.db",
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

        print("INCOMING:", body)


        conversation_id = (
            body.get("conversation_id")
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



        # -------------------------
        # Load memory
        # -------------------------

        cursor.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id=?
            ORDER BY id ASC
            """,
            (conversation_id,)
        )


        old_messages = cursor.fetchall()



        messages = [
            {
                "role": "system",
                "content":
                "You are Surf AI, a friendly helpful assistant. Remember details from the conversation."
            }
        ]


        for role, content in old_messages:
            messages.append(
                {
                    "role": role,
                    "content": content
                }
            )



        # -------------------------
        # Save user message
        # -------------------------

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id,user_id,role,content)
            VALUES (?,?,?,?)
            """,
            (
                conversation_id,
                user_id,
                "user",
                message
            )
        )

        db.commit()



        messages.append(
            {
                "role": "user",
                "content": message
            }
        )



        # -------------------------
        # OpenAI
        # -------------------------

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )


        reply = response.choices[0].message.content



        # -------------------------
        # Save AI reply
        # -------------------------

        cursor.execute(
            """
            INSERT INTO messages
            (conversation_id,user_id,role,content)
            VALUES (?,?,?,?)
            """,
            (
                conversation_id,
                user_id,
                "assistant",
                reply
            )
        )

        db.commit()



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
            {
                "role": role,
                "content": content
            }
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


    return {
        "status": "deleted"
    }

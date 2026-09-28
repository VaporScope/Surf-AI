import os
import uuid
import sqlite3
import base64

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

        # Base44 compatibility fallback check
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

        # Grab optional file data sent from the frontend
        # Expected structure: {"name": "test.png", "type": "image/png", "base64": "..."}
        file_data = body.get("file")

        if not message and not file_data:
            return {
                "error": "Empty message and no files provided",
                "conversation_id": conversation_id
            }

        # 1. Load previous memory (Limit to last 20 messages to prevent token limits)
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

        # 2. Construct System Prompt Context with your custom identity
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Surf AI, a friendly and helpful assistant. "
                    "You were created by an aerospace engineering and AI/ML student at Université Paris-Saclay. "
                    "If a user asks who made you, who created you, or about your developer, always state clearly that you were built by this student at Université Paris-Saclay. "
                    "You can inspect uploaded images or read text contents from uploaded documents. "
                    "Always refer to the user's provided context files or images if present. "
                    "Remember details from earlier in the conversation history to answer follow-up prompts smoothly."
                )
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        # 3. Process the incoming content structure for OpenAI (Handles Text + Images)
        current_content = []
        db_log_content = message  # Fallback text to store cleanly in your SQLite database logs

        if message:
            current_content.append({"type": "text", "text": message})

        if file_data:
            file_name = file_data.get("name", "uploaded_file")
            file_type = file_data.get("type", "")
            base64_str = file_data.get("base64", "")

            # Case A: File is an image (Leverage GPT-4o-mini vision capabilities)
            if "image" in file_type:
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{file_type};base64,{base64_str}"}
                })
                db_log_content += f" [Sent Image: {file_name}]"
            
            # Case B: File is a text-based document (.txt, .md, .csv, .json, etc.)
            else:
                try:
                    # Decode base64 bytes string back into clean readable text context
                    decoded_text = base64.b64decode(base64_str).decode("utf-8")
                    file_context = f"\n\n[Attached File Content for reference from '{file_name}']:\n{decoded_text}"
                    current_content.append({"type": "text", "text": file_context})
                    db_log_content += f" [Attached Document: {file_name}]"
                except Exception:
                    # Fallback string if binary properties can't be string-parsed
                    current_content.append({"type": "text", "text": f"\n\n[Attached unreadable file placeholder: {file_name}]"})

        # 4. Save incoming user message to SQLite memory database
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, user_id, "user", db_log_content)
        )
        db.commit()

        # Add current structured payload layer onto runtime array
        messages.append({"role": "user", "content": current_content})

        # 5. Send complete thread to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = response.choices[0].message.content

        # 6. Save AI reply to SQLite memory database
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, user_id, "assistant", reply)
        )
        db.commit()

        # Return response to Base44
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

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

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API KEYS & CLIENTS
# =========================
api_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("OPENAI KEY EXISTS:", bool(api_key))
print("WEB SEARCH KEY EXISTS:", bool(tavily_key))

client = OpenAI(
    api_key=api_key,
    timeout=60
)

# =========================
# DATABASE
# =========================
DB_PATH = "/data/surf_memory.db" if os.path.exists("/data") else "surf_memory.db"

db = sqlite3.connect(DB_PATH, check_same_thread=False)
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
# HELPER: LIVE WEB SEARCH FUNCTION
# =========================
def search_the_web(query: str) -> str:
    """Queries Tavily API to fetch live search engine text data and reference URLs."""
    if not tavily_key:
        return "[System Notice: Live web search is currently unavailable because TAVILY_API_KEY is missing.]"
    
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
                # Provide index mappings clearly so the AI knows which number points to which real URL
                context_string += f"Index: [{i}]\nSource Name: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}\n\n"
            
            return context_string
        return "[System Notice: Web search failed to respond.]"
    except Exception as e:
        return f"[System Notice: Web search error occurred: {str(e)}]"

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
        location_data = body.get("location")

        if not message and not file_data:
            return {
                "error": "Empty message and no files provided",
                "conversation_id": conversation_id
            }

        # Auto-register new chat titles
        cursor.execute(
            "INSERT OR IGNORE INTO conversations (conversation_id, title) VALUES (?, ?)",
            (conversation_id, message[:30] if message else "New Chat")
        )
        db.commit()

        # Load chat history
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
            (conversation_id,)
        )
        old_messages = cursor.fetchall()[::-1]

        # 🔒 Privacy & Inline Citation System Instructions
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Surf AI, a friendly and helpful assistant created by an aerospace engineering and AI/ML student at Université Paris-Saclay. "
                    "CRITICAL CORE FOCUS: This application places a very high emphasis on user privacy. "
                    "Reassure users politely about data security if they ask. "
                    "WEB SEARCH CAPABILITY: You have a live connection to search engines. "
                    "Below the user's prompt, you will see a section titled '--- LIVE WEB SEARCH RESULTS ---'. "
                    "Use this live data to answer the user accurately with up-to-date facts. "
                    "CITATION REQUIREMENT: When using facts from the search results, you MUST cite them inline using clean standard markdown hyperlink formats "
                    "pointing directly to their source URL string. Format them exactly as a tiny numbered link, for example: [1](URL), [2](URL), etc. "
                    "DO NOT create a separate Bibliography, Sources list, or reference index at the bottom of the message. Keep everything inline as you type your response."
                )
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        current_content = []
        db_log_content = message 

        # 1. Trigger live web search
        if message:
            search_data = search_the_web(message)
            current_content.append({"type": "text", "text": search_data})

        # 2. Inject location context if authorized
        if location_data:
            lat = location_data.get("latitude")
            lon = location_data.get("longitude")
            current_content.append({
                "type": "text", 
                "text": f"\n[User Current Location: Lat {lat}, Lon {lon}]\n"
            })

        if message:
            current_content.append({"type": "text", "text": f"User Question: {message}"})

        # 3. Handle File uploads
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
                    file_context = f"\n\n[Attached File Content '{file_name}']:\n{decoded_text}"
                    current_content.append({"type": "text", "text": file_context})
                    db_log_content += f" [Attached Document: {file_name}]"
                except Exception:
                    current_content.append({"type": "text", "text": f"\n\n[Attached unreadable file placeholder: {file_name}]"})

        # Record chat log
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

        # Generate live answer via OpenAI using search context
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = response.choices.message.content

        # Record assistant reply
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
# GET ALL CONVERSATIONS LIST
# =========================
@app.get("/history")
def get_all_conversations():
    try:
        cursor.execute("SELECT conversation_id, title FROM conversations ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [{"conversation_id": row[0], "title": row[1]} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# RENAME CONVERSATION
# =========================
@app.put("/history/{conversation_id}/rename")
def rename_conversation(conversation_id: str, request: RenameRequest):
    try:
        cursor.execute("UPDATE conversations SET title = ? WHERE conversation_id = ?", (request.title, conversation_id))
        db.commit()
        return {"status": "success", "message": f"Conversation renamed to {request.title}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# GET SINGLE CHAT HISTORY
# =========================
@app.get("/history/{conversation_id}")
def get_history(conversation_id):
    cursor.execute("SELECT title FROM conversations WHERE conversation_id=?", (conversation_id,))
    title_row = cursor.fetchone()

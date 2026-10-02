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

# Enable global Cross-Origin Resource Sharing (CORS) for smooth frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global environment api access credentials
api_key = os.getenv("OPENAI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

print("OPENAI KEY EXISTS:", bool(api_key))
print("SERPER GOOGLE SEARCH KEY EXISTS:", bool(serper_key))

client = OpenAI(
    api_key=api_key,
    timeout=60
)

# Persistent volume file pathway setup for database instances
DB_PATH = "/data/surf_memory.db" if os.path.exists("/data") else "surf_memory.db"

db = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)
cursor = db.cursor()

# Initialize data schemas
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


# Validation schemas
class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str | None = None
    user_id: str | None = None


class RenameRequest(BaseModel):
    title: str


def search_the_web(query: str) -> str:
    if not serper_key:
        return "[System Notice: Live Google web search is currently unavailable because SERPER_API_KEY is missing on Render.]"
    try:
        # FIXED: correct Serper search endpoint
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": serper_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "gl": "fr",  # France regional search localization
            "hl": "en"   # Preferred language
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get("organic", [])
            if not results or not isinstance(results, list):
                return "[System Notice: The search completed successfully but returned zero relevant organic search match results.]"

            context_string = "\n--- LIVE WEB SEARCH RESULTS ---\n"
            for i, res in enumerate(results[:5], 1):
                context_string += (
                    f"Source Mapping Index: [{i}]\n"
                    f"Source Name: {res.get('title')}\n"
                    f"URL: {res.get('link')}\n"
                    f"Content Summary: {res.get('snippet')}\n\n"
                )
            return context_string
        return f"[System Notice: Google web search API failed to respond properly. Status Code: {response.status_code}]"
    except Exception as e:
        return f"[System Notice: Google web search exception occurred: {str(e)}]"


@app.get("/")
def home():
    return {"status": "Surf AI is running"}


@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("INCOMING INTERFACE ACTION:", body)

        # Unify various frontend request session conventions
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

        # Initialize tracking index for conversation
        cursor.execute(
            "INSERT OR IGNORE INTO conversations (conversation_id, title) VALUES (?, ?)",
            (conversation_id, message[:30] if message else "New Chat")
        )
        db.commit()

        # Reconstruct recent history (last 20 messages)
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
                    "If a user asks who made you, always state clearly that you are built by this specific student at Université Paris-Saclay. "
                    "PRIVACY: This project places a strong emphasis on user privacy. "
                    "If users ask about data tracking or message logging, be honest: conversations are stored so that chat history works, "
                    "and they are not used for advertising or sold to third parties. "
                    "REAL-TIME SEARCH CAPABILITIES: You have a live connection to Google search. "
                    "Data containing live updates or local places context will be embedded below the user prompt under '--- LIVE WEB SEARCH RESULTS ---'. "
                    "STYLE RULES: Never say things like 'I cannot access real-time location data', "
                    "'I cannot browse the internet', or recommend external services like Google Maps, Yelp, or Uber Eats. "
                    "If search data is absent or location parsing fails, simply use your general knowledge to answer, or ask the user directly which city, "
                    "neighborhood, or commune they are currently looking in so you can guide them. "
                    "COMPACT INLINE CITATIONS ONLY: Cite search data immediately inline using markdown links where the "
                    "visible text is a numbered reference matching the index value, formatted precisely like [1](URL) or [2](URL). "
                    "Never output a bibliography block or sources section at the bottom of the response."
                )
            }
        ]

        for role, content in old_messages:
            messages.append({"role": role, "content": content})

        current_content = []
        db_log_content = message

        search_query = message
        if location_data:
            lat = location_data.get("latitude")
            lon = location_data.get("longitude")
            search_query += f" near coordinates {lat}, {lon}"
            current_content.append({
                "type": "text",
                "text": f"\n[System Log: User granted location access. Approximate Coordinates: Lat {lat}, Lon {lon}]\n"
            })

        if message:
            search_data = search_the_web(search_query)
            current_content.append({"type": "text", "text": search_data})

        if message:
            current_content.append({"type": "text", "text": f"User Prompt: {message}"})

        # Process images or documents
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

        # Record user message
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
        # FIXED: choices is a list
        reply = response.choices[0].message.content

        # Record assistant reply
        cursor.execute(
            "INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, "assistant", reply)
        )
        db.commit()

        return {
            "conversation_id": conversation_id,
            "reply": reply
        }

    except Exception as e:
        print(f"Error occurred in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    try:
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,)
        )
        rows = cursor.fetchall()
        history = [{"role": r, "content": c} for r, c in rows]
        return {"conversation_id": conversation_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rename/{conversation_id}")
async def rename_conversation(conversation_id: str, request: RenameRequest):
    try:
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
            (request.title, conversation_id)
        )
        db.commit()
        return {"status": "success", "conversation_id": conversation_id, "new_title": request.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import re
import json
import time
import uuid
import socket
import hashlib
import sqlite3
import base64
import ipaddress
import threading
import requests
from collections import defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Surf AI API")

# ------------------------------------------------------------------ Settings
api_key = os.getenv("OPENAI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = 3

# AUTH_MODE: "off"    = ignore tokens (rate limits still apply)
#            "soft"   = verify tokens when sent, but still accept unverified requests (rollout)
#            "strict" = real accounts REQUIRE a valid token
AUTH_MODE = os.getenv("AUTH_MODE", "soft").lower()
AUTH_VERIFY_URL = os.getenv("AUTH_VERIFY_URL", "")

IP_PER_MIN = int(os.getenv("IP_PER_MIN", "12"))          # chat requests per IP per minute
GUEST_PER_DAY = int(os.getenv("GUEST_PER_DAY", "30"))    # chat requests per guest per day
USER_PER_DAY = int(os.getenv("USER_PER_DAY", "200"))     # chat requests per logged-in user per day
API_PER_MIN = int(os.getenv("API_PER_MIN", "120"))       # all other endpoints per IP per minute
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
MAX_FILE_B64 = int(os.getenv("MAX_FILE_B64", "8000000"))  # ~6 MB file
GUEST_UPLOADS_PER_DAY = int(os.getenv("GUEST_UPLOADS_PER_DAY", "3"))
USER_UPLOADS_PER_DAY = int(os.getenv("USER_UPLOADS_PER_DAY", "15"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"https://([a-z0-9-]+\.)*(asksurf\.uk|base44\.app|base44\.com)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

print("OPENAI KEY EXISTS:", bool(api_key))
print("SERPER KEY EXISTS:", bool(serper_key))
print("MODEL:", MODEL)
print("AUTH_MODE:", AUTH_MODE, "| VERIFY URL SET:", bool(AUTH_VERIFY_URL))

client = OpenAI(api_key=api_key, timeout=60)

# ---------------------------------------------------------------- Database
DB_PATH = "/data/surf_memory.db" if os.path.exists("/data") else "surf_memory.db"
print("DB PATH:", DB_PATH)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db_lock = threading.Lock()


def q(sql, params=()):
    """Run one SQL statement safely from any thread and commit."""
    with db_lock:
        cur = db.execute(sql, params)
        rows = cur.fetchall()
        db.commit()
        return rows


q("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    user_id TEXT,
    role TEXT,
    content TEXT
)
""")
q("""
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
try:
    q("ALTER TABLE conversations ADD COLUMN user_id TEXT")
except sqlite3.OperationalError:
    pass  # column already exists

q("""
UPDATE conversations
SET user_id = (
    SELECT m.user_id FROM messages m
    WHERE m.conversation_id = conversations.conversation_id
    LIMIT 1
)
WHERE user_id IS NULL
""")
q("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id)")
q("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id)")


class RenameRequest(BaseModel):
    title: str


# ------------------------------------------------------------ Rate limiting
_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def hit(key: str, limit: int, window: int) -> int:
    """Record a hit. Returns 0 if allowed, otherwise seconds to wait."""
    now = time.time()
    with _hits_lock:
        if len(_hits) > 20000:  # housekeeping
            for k in [k for k, v in _hits.items() if not v or v[-1] < now - 86400]:
                del _hits[k]
        dq = _hits[key]
        while dq and dq[0] <= now - window:
            dq.popleft()
        if len(dq) >= limit:
            return int(window - (now - dq[0])) + 1
        dq.append(now)
        return 0


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def too_many(retry: int, msg: str):
    raise HTTPException(status_code=429, detail=msg, headers={"Retry-After": str(retry)})


def light_limit(request: Request):
    retry = hit(f"api:{client_ip(request)}", API_PER_MIN, 60)
    if retry:
        too_many(retry, "Too many requests. Please slow down.")


# ------------------------------------------------------------ Identity/auth
GUEST_RE = re.compile(r"^guest-[A-Za-z0-9-]{16,64}$")
_token_cache = {}  # sha256(token) -> (user_id, expires_at)
_token_lock = threading.Lock()


def verify_token(token: str):
    """Ask the verify endpoint who this token belongs to. Returns user id or None."""
    if not token or not AUTH_VERIFY_URL:
        return None
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _token_lock:
        cached = _token_cache.get(key)
        if cached and cached[1] > now:
            return cached[0]
        if len(_token_cache) > 5000:
            _token_cache.clear()
    try:
        r = requests.post(
            AUTH_VERIFY_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={},
            timeout=6,
        )
        if r.status_code != 200:
            print(f"AUTH: verify returned {r.status_code}")
            return None
        data = r.json()
        user = data.get("user") if isinstance(data.get("user"), dict) else data
        uid = user.get("id") or user.get("_id")
        if not uid:
            print("AUTH: verify response had no id")
            return None
        uid = str(uid)
        with _token_lock:
            _token_cache[key] = (uid, now + 300)  # cache 5 minutes
        return uid
    except Exception as e:
        print(f"AUTH: verify error: {e}")
        return None


def identify(request: Request, claimed):
    """Return (identity, is_guest)."""
    claimed = (claimed or "").strip()
    token = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()

    verified = verify_token(token) if (token and AUTH_MODE != "off") else None
    if verified:
        return verified, False

    is_guest_id = bool(GUEST_RE.match(claimed))
    is_anon = claimed in ("", "guest")

    if AUTH_MODE == "strict":
        if is_guest_id:
            return claimed, True
        if is_anon:
            return "guest", True
        raise HTTPException(status_code=401, detail="Please sign in again.")

    # off / soft: accept the claimed id (rollout period)
    if is_guest_id or is_anon:
        return (claimed or "guest"), True
    return claimed, False


def owner_of(conversation_id: str):
    rows = q("SELECT user_id FROM conversations WHERE conversation_id = ?", (conversation_id,))
    return rows[0][0] if rows else None


def require_owner(conversation_id: str, identity: str):
    """Read/rename/delete access: the conversation must belong to the caller."""
    owner = owner_of(conversation_id)
    if owner is None:
        return
    if identity == "guest" and AUTH_MODE != "strict":
        return  # rollout: older frontend does not send user_id on every call yet
    if identity == "guest" or owner != identity:
        raise HTTPException(status_code=404, detail="Conversation not found")


def prepare_chat(request: Request, body: dict) -> dict:
    ip = client_ip(request)
    retry = hit(f"chat-min:{ip}", IP_PER_MIN, 60)
    if retry:
        too_many(retry, "You're sending messages too fast. Please wait a moment.")

    identity, is_guest = identify(request, body.get("user_id"))
    day_key = f"chat-day:{identity}" + (f":{ip}" if identity == "guest" else "")
    retry = hit(day_key, GUEST_PER_DAY if is_guest else USER_PER_DAY, 86400)
    if retry:
        msg = ("Daily message limit reached for guests. Sign in for a higher limit."
               if is_guest else "Daily message limit reached. Please try again tomorrow.")
        too_many(retry, msg)

    message = body.get("message") or body.get("content") or ""
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message is too long.")

    f = body.get("file")
    if f:
        if len(f.get("base64", "")) > MAX_FILE_B64:
            raise HTTPException(status_code=413, detail="File is too large.")
        ftype = (f.get("type") or "").lower()
        if "image" in ftype and ftype not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type. Use JPG, PNG, WebP or GIF.")
        # Guests are limited by IP so clearing browser data doesn't reset the count
        up_key = f"upload-day:ip:{ip}" if is_guest else f"upload-day:{identity}"
        retry = hit(up_key, GUEST_UPLOADS_PER_DAY if is_guest else USER_UPLOADS_PER_DAY, 86400)
        if retry:
            too_many(retry, "Daily upload limit reached. Sign in for more uploads."
                     if is_guest else "Daily upload limit reached. Please try again tomorrow.")

    cid = body.get("conversation_id") or body.get("session_id") or body.get("chat_id")
    if cid:
        owner = owner_of(cid)
        if owner is not None and owner != identity:
            raise HTTPException(status_code=403, detail="This conversation belongs to someone else.")

    body["user_id"] = identity
    return body


# ------------------------------------------------------------------- Tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search Google for current or factual information: news, scores, prices, "
                "weather, recent events, local places, products, or anything that may have "
                "changed recently. Do NOT use for greetings, writing, math, coding, or "
                "stable general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short, specific search query."},
                    "search_type": {
                        "type": "string",
                        "enum": ["web", "news", "places"],
                        "description": "web = general (default), news = recent news, places = local businesses/locations.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Open a web page and read its text. Use after a search when snippets are not "
                "enough, or when the user gives a URL. Use at most 2 pages per question."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full http(s) URL."}},
                "required": ["url"],
            },
        },
    },
]


def is_safe_url(url: str) -> bool:
    """Block non-http URLs and private/internal addresses."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
        return True
    except Exception:
        return False


def search_the_web(query: str, search_type: str = "web", state: dict = None) -> str:
    if state is None:
        state = {"n": 0}
    if not query:
        return "[Notice: empty search query.]"
    if not serper_key:
        return "[Notice: web search is unavailable because SERPER_API_KEY is missing.]"
    endpoint = {"web": "search", "news": "news", "places": "places"}.get(search_type, "search")
    try:
        r = requests.post(
            f"https://google.serper.dev/{endpoint}",
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            json={"q": query, "gl": "fr", "hl": "en"},
            timeout=10,
        )
        if r.status_code != 200:
            return f"[Notice: search failed with status {r.status_code}.]"
        data = r.json()
        out = []

        box = data.get("answerBox")
        if isinstance(box, dict):
            ans = box.get("answer") or box.get("snippet")
            if ans:
                out.append(f"DIRECT ANSWER (Google): {ans}")

        kg = data.get("knowledgeGraph")
        if isinstance(kg, dict) and kg.get("title"):
            line = f"KNOWLEDGE PANEL: {kg.get('title')}"
            if kg.get("type"):
                line += f" ({kg['type']})"
            if kg.get("description"):
                line += f": {kg['description']}"
            attrs = kg.get("attributes")
            if isinstance(attrs, dict):
                line += " | " + "; ".join(f"{k}: {v}" for k, v in list(attrs.items())[:6])
            out.append(line)

        if endpoint == "search":
            for res in (data.get("organic") or [])[:5]:
                state["n"] += 1
                out.append(
                    f"[{state['n']}] {res.get('title')}\nURL: {res.get('link')}\n"
                    f"{res.get('date', '')} {res.get('snippet', '')}".strip()
                )
        elif endpoint == "news":
            for res in (data.get("news") or [])[:5]:
                state["n"] += 1
                out.append(
                    f"[{state['n']}] {res.get('title')} ({res.get('source', '')}, {res.get('date', '')})\n"
                    f"URL: {res.get('link')}\n{res.get('snippet', '')}"
                )
        else:
            for res in (data.get("places") or [])[:5]:
                state["n"] += 1
                out.append(
                    f"[{state['n']}] {res.get('title')} - {res.get('category', '')}\n"
                    f"Address: {res.get('address', '')} | Rating: {res.get('rating', 'n/a')} "
                    f"({res.get('ratingCount', 0)} reviews)\n"
                    f"URL: {res.get('website', '')}"
                )

        if not out:
            return "[Notice: the search returned no useful results.]"
        return "\n\n".join(out)
    except Exception as e:
        return f"[Notice: search error: {e}]"


def read_page(url: str) -> str:
    """Fetch a page and return cleaned text (max ~5000 chars)."""
    r = None
    try:
        current = url
        for _ in range(4):
            if not is_safe_url(current):
                return "[Notice: that URL cannot be opened.]"
            r = requests.get(
                current,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SurfAI/1.0)"},
                timeout=8,
                stream=True,
                allow_redirects=False,
            )
            if r.status_code in (301, 302, 303, 307, 308):
                current = urljoin(current, r.headers.get("Location", ""))
                r.close()
                continue
            break
        else:
            return "[Notice: too many redirects.]"

        if r.status_code != 200:
            return f"[Notice: the page returned status {r.status_code}.]"
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype:
            return "[Notice: that page is not readable text.]"

        raw = r.raw.read(1_000_000, decode_content=True)
        html = raw.decode(r.encoding or "utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "form", "noscript", "svg"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        if not text:
            return "[Notice: no readable text found on that page.]"
        return f"PAGE CONTENT ({current}):\n{text[:5000]}"
    except Exception as e:
        return f"[Notice: could not read the page: {e}]"
    finally:
        if r is not None:
            r.close()


# ------------------------------------------------------------ System prompt
def build_system_prompt(user_name, location_data, web_search, response_style, custom_instructions):
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    p = (
        "You are Surf AI, a friendly and helpful assistant created by an"
        "and AI/ML student at Aivancity. If a user asks who made you, say clearly "
        "that you were built by this student at Aivancity. "
        "PRIVACY: this project places a strong emphasis on user privacy. If users ask about data "
        "tracking or logging, be honest: conversations are stored so chat history works, and they "
        "are not used for advertising or sold to third parties. "
        f"Today's date is {today} (UTC). "
    )
    if user_name:
        p += f"The user's name is {user_name}. "
    if location_data:
        lat, lon = location_data.get("latitude"), location_data.get("longitude")
        if lat is not None and lon is not None:
            p += (f"The user's approximate coordinates are {lat}, {lon}. For local questions, "
                  "work out the nearby city or area and include its name in search queries. ")
    if web_search:
        p += (
            "You have tools: search_web and read_page. Use search_web only when the question needs "
            "current or verifiable information (news, scores, prices, weather, local places, "
            "recent releases, anything that may have changed). Do not search for greetings, "
            "writing, math, coding, or stable general knowledge. Keep queries short. If a search "
            "returns a DIRECT ANSWER or KNOWLEDGE PANEL, prefer it. If snippets are not enough, "
            "use read_page on the best 1-2 results. Never say you cannot browse the internet. "
            "CITATIONS: cite sources inline as markdown links whose visible text is the source "
            "number, like [1](URL) or [2](URL), using the numbers from the search results. "
            "Never add a sources list at the end. "
        )
    else:
        p += (
            "Live web search is turned off in the user's settings. For questions needing current "
            "information, answer from general knowledge, say it may be out of date, and mention "
            "they can turn on web search in Settings > Data & Privacy. "
        )
    p += f"RESPONSE STYLE: {response_style}. "
    if custom_instructions:
        p += f"USER CUSTOM INSTRUCTIONS: {custom_instructions}"
    return p


# --------------------------------------------------------------- Chat engine
def chat_events(body: dict):
    """Generator that yields event dicts: meta, status, token, done, error."""
    try:
        conversation_id = (
            body.get("conversation_id") or body.get("session_id")
            or body.get("chat_id") or str(uuid.uuid4())
        )
        message = (body.get("message") or body.get("content") or "").strip()
        user_id = body.get("user_id") or "guest"
        user_name = (body.get("user_name") or "").strip()[:50]
        file_data = body.get("file")
        location_data = body.get("location")
        custom_instructions = (body.get("custom_instructions") or "")[:1000]
        response_style = body.get("response_style") or "Balanced"
        web_search = body.get("web_search", True) not in (False, "false", 0)

        if not message and not file_data:
            yield {"type": "error", "text": "Empty message and no files provided", "status": 400}
            return

        yield {"type": "meta", "conversation_id": conversation_id}

        q("INSERT OR IGNORE INTO conversations (conversation_id, user_id, title) VALUES (?, ?, ?)",
          (conversation_id, user_id, message[:30] if message else "New Chat"))

        old = q("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
                (conversation_id,))[::-1]

        messages = [{"role": "system", "content": build_system_prompt(
            user_name, location_data, web_search, response_style, custom_instructions)}]
        for role, content in old:
            messages.append({"role": role, "content": content})

        parts = [{"type": "text", "text": message or "Please look at the attached file."}]
        db_log = message

        if file_data:
            file_name = file_data.get("name", "uploaded_file")
            file_type = file_data.get("type", "")
            b64 = file_data.get("base64", "")
            if "image" in file_type:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{file_type};base64,{b64}", "detail": "auto"}})
                db_log += f" [Attached Image: {file_name}]"
            else:
                try:
                    text = base64.b64decode(b64).decode("utf-8")[:20000]
                    parts.append({"type": "text",
                                  "text": f"\n\n[Contents of Document '{file_name}']:\n{text}"})
                    db_log += f" [Attached Text Document: {file_name}]"
                except Exception:
                    parts.append({"type": "text",
                                  "text": f"\n\n[Attached Unreadable File: {file_name}]"})

        q("INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
          (conversation_id, user_id, "user", db_log))
        q("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
          (conversation_id,))
        messages.append({"role": "user", "content": parts})

        state = {"n": 0}
        full_reply = ""

        for round_no in range(MAX_TOOL_ROUNDS + 1):
            kwargs = {"model": MODEL, "messages": messages, "stream": True}
            if web_search and round_no < MAX_TOOL_ROUNDS:
                kwargs["tools"] = TOOLS

            stream = client.chat.completions.create(**kwargs)
            round_text = ""
            calls = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    round_text += delta.content
                    full_reply += delta.content
                    yield {"type": "token", "text": delta.content}
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            c["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                c["name"] += tc.function.name
                            if tc.function.arguments:
                                c["args"] += tc.function.arguments

            if not calls:
                break

            ordered = [c for _, c in sorted(calls.items())]
            messages.append({
                "role": "assistant",
                "content": round_text or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"]}}
                    for c in ordered
                ],
            })

            for c in ordered:
                try:
                    args = json.loads(c["args"] or "{}")
                except Exception:
                    args = {}
                if c["name"] == "search_web":
                    yield {"type": "status", "text": f"Searching the web for: {args.get('query', '')}"}
                    result = search_the_web(args.get("query", ""), args.get("search_type", "web"), state)
                elif c["name"] == "read_page":
                    yield {"type": "status", "text": "Reading a page..."}
                    result = read_page(args.get("url", ""))
                else:
                    result = "[Notice: unknown tool.]"
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        if not full_reply.strip():
            full_reply = "Sorry, I couldn't generate a reply. Please try again."
            yield {"type": "token", "text": full_reply}

        q("INSERT INTO messages (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
          (conversation_id, user_id, "assistant", full_reply))
        q("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
          (conversation_id,))
        yield {"type": "done", "conversation_id": conversation_id}

    except Exception as e:
        print(f"Chat error: {e}")
        yield {"type": "error", "text": "Something went wrong. Please try again.", "status": 500}


# ---------------------------------------------------------------- Endpoints
@app.get("/")
def home():
    return {"status": "Surf AI is running"}


@app.post("/chat/stream")
def chat_stream(request: Request, body: dict = Body(...)):
    body = prepare_chat(request, body)

    def gen():
        for ev in chat_events(body):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
def chat(request: Request, body: dict = Body(...)):
    """Non-streaming version (same engine). Kept for compatibility."""
    body = prepare_chat(request, body)
    reply, cid = "", None
    for ev in chat_events(body):
        if ev["type"] == "meta":
            cid = ev["conversation_id"]
        elif ev["type"] == "token":
            reply += ev["text"]
        elif ev["type"] == "error":
            raise HTTPException(status_code=ev.get("status", 500), detail=ev["text"])
    return {"conversation_id": cid, "reply": reply}


@app.get("/history/{conversation_id}")
def get_chat_history(conversation_id: str, request: Request, user_id: str = ""):
    light_limit(request)
    identity, _ = identify(request, user_id)
    require_owner(conversation_id, identity)
    try:
        rows = q("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id ASC",
                 (conversation_id,))
        return {"conversation_id": conversation_id,
                "history": [{"role": r, "content": c} for r, c in rows]}
    except Exception as e:
        print(f"History error: {e}")
        raise HTTPException(status_code=500, detail="Could not load history.")


@app.get("/conversations")
def list_conversations(request: Request, user_id: str = ""):
    light_limit(request)
    identity, _ = identify(request, user_id)
    if identity == "guest":
        return {"conversations": []}
    try:
        rows = q("SELECT conversation_id, title, updated_at FROM conversations "
                 "WHERE user_id = ? ORDER BY updated_at DESC", (identity,))
        return {"conversations": [
            {"conversation_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]}
    except Exception as e:
        print(f"List error: {e}")
        raise HTTPException(status_code=500, detail="Could not load conversations.")


@app.post("/rename/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenameRequest,
                        request: Request, user_id: str = ""):
    light_limit(request)
    identity, _ = identify(request, user_id)
    require_owner(conversation_id, identity)
    title = payload.title.strip()[:100] or "Untitled"
    try:
        q("UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
          (title, conversation_id))
        return {"status": "success", "conversation_id": conversation_id, "new_title": title}
    except Exception as e:
        print(f"Rename error: {e}")
        raise HTTPException(status_code=500, detail="Could not rename.")


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request, user_id: str = ""):
    light_limit(request)
    identity, _ = identify(request, user_id)
    require_owner(conversation_id, identity)
    try:
        q("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        q("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        return {"status": "deleted", "conversation_id": conversation_id}
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete.")


@app.delete("/user/{user_id}/data")
def delete_user_data(user_id: str, request: Request):
    light_limit(request)
    identity, _ = identify(request, user_id)
    if identity == "guest" or identity != user_id:
        raise HTTPException(status_code=403, detail="Not allowed.")
    try:
        q("DELETE FROM messages WHERE user_id = ?", (user_id,))
        q("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        return {"status": "deleted", "user_id": user_id}
    except Exception as e:
        print(f"Delete user data error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete data.")

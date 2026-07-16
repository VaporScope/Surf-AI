import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Chat(BaseModel):
    message: str
    
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <style>
.user {
  background: #007bff;
  color: white;
  padding: 10px;
  border-radius: 10px;
  margin: 5px;
  text-align: right;
}

.bot {
  background: #f1f1f1;
  padding: 10px;
  border-radius: 10px;
  margin: 5px;
  text-align: left;
}
</style>
    <body style="font-family:Arial; text-align:center; margin-top:50px;">
    <div style="max-width:500px; margin:auto; margin-top:50px;">
        <img src="/logo" width="250" style="margin-bottom:15px;">
        <input id="msg" placeholder="Type message"
style="width:300px; height:40px; font-size:16px; padding:5px;">
        <button onclick="send()"
style="height:40px; font-size:16px; padding:5px 15px;">
Send</button>
        <div id="chat" style="
height:300px;
overflow-y:auto;
border:1px solid #ccc;
padding:10px;
border-radius:10px;
margin-top:20px;
text-align:left;">
</div>

        <script>
async function send() {
    let msg = document.getElementById("msg").value;
    if (!msg) return;

    let chat = document.getElementById("chat");

    // USER MESSAGE
    chat.innerHTML += `<div class="user">You: ${msg}</div>`;

    // CLEAR INPUT
    document.getElementById("msg").value = "";

    // TYPING ANIMATION
    let typingId = "typing";
    chat.innerHTML += `<div id="${typingId}" class="bot">Surf AI is typing...</div>`;

    chat.scrollTop = chat.scrollHeight;

    // SEND TO BACKEND
    let res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: msg})
    });

    let data = await res.json();

    // REMOVE TYPING
    document.getElementById(typingId).remove();

    // BOT RESPONSE
    chat.innerHTML += `<div class="bot">Surf AI: ${data.reply}</div>`;

    chat.scrollTop = chat.scrollHeight;
}
</script>
<div id="chat" style="
height:300px;
overflow-y:auto;
border:1px solid #ccc;
padding:10px;
border-radius:10px;
margin-top:20px;
text-align:left;">
</div>
        </div>
    </body>
    </html>
    """
    
@app.get("/logo")
def get_logo():
    return FileResponse("logo.png")

current_folder = Path(__file__).parent
gif_file = current_folder / "surfer.gif"


@app.get("/surfer.gif")
async def get_gif():
    if gif_file.exists():
        return FileResponse(gif_file, media_type="image/gif")
    return HTMLResponse("GIF not found!", status_code=404)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>Surf AI Page</title>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
            }
            img.fullscreen-gif {
                width: 100%;
                height: 100%;
                object-fit: cover; /* cover entire page */
            }
        </style>
    </head>
    <body>
        <img class="fullscreen-gif" src="/surfer.gif" alt="Surf GIF">
    </body>
    </html>
    """
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>Surf AI Page</title>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
            }
            img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <img src="/static/surfer.gif" alt="Surfer GIF">
    </body>
    </html>
    """


@app.post("/chat")
def chat_ai(chat: Chat):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Surf, a friendly AI chatbot."},
            {"role": "user", "content": chat.message}
        ]
    )
    return {"reply": response.choices[0].message.content}
from fastapi.responses import FileResponse







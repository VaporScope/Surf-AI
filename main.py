import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI(title="Surf AI API")


# Allow Base44 frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict this to your Base44 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI connection
api_key = os.getenv("OPENAI_API_KEY")

print("OPENAI KEY EXISTS:", bool(api_key))

client = OpenAI(
    api_key=api_key,
    timeout=60.0
)

# Request format from Base44
class ChatRequest(BaseModel):
    message: str


# Test endpoint
@app.get("/")
def home():
    return {
        "status": "Surf AI is running"
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are Surf AI, a friendly helpful assistant."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ]
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        print("OPENAI ERROR:", str(e))
        return {
            "error": str(e)
        }







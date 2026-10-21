import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pobieramy token ze zmiennej środowiskowej serwera
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Zabezpieczenie na przypadek, gdyby zmienna nie została ustawiona
if not GITHUB_TOKEN:
    raise RuntimeError("Brak zmiennej środowiskowej GITHUB_TOKEN!")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)

class HeadlineInput(BaseModel):
    title: str

@app.post("/analyze")
async def analyze_headline(input_data: HeadlineInput):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś analitykiem NLP. Przeanalizuj nagłówek pod kątem clickbaitu (0-100%) i ładunku emocjonalnego (1-10). Odpowiedz WYŁĄCZNIE czystym JSONem w formacie: {\"clickbait\": int, \"emotion\": int}"
                },
                {
                    "role": "user",
                    "content": input_data.title
                }
            ],
            model="Meta-Llama-3-8B-Instruct",
            response_format={"type": "json_object"}
        )
        
        ai_response = chat_completion.choices[0].message.content
        return json.loads(ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

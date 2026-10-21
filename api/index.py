import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class HeadlineBatchInput(BaseModel):
    titles: List[str]

@app.get("/")
def read_root():
    return {"status": "News Guard API (Groq) dziala!"}

@app.post("/analyze-batch")
async def analyze_batch(input_data: HeadlineBatchInput):
    # Bezpieczne pobranie klucza wewnątrz funkcji
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="Błąd: Brak zmiennej GROQ_API_KEY w panelu Vercel!")

    try:
        client = Groq(api_key=groq_api_key)
        formatted_titles = "\n".join([f"{idx + 1}. {title}" for idx, title in enumerate(input_data.titles)])

        system_prompt = (
            "Jesteś analitykiem NLP. Przeanalizuj poniższą listę nagłówków prasowych. "
            "Dla każdego nagłówka określ poziom clickbaitu (0-100%) oraz ładunek emocjonalny (1-10). "
            "Zwróć WYŁĄCZNIE czysty obiekt JSON z kluczem 'results', zawierającym tablicę obiektów. "
            "Przykład formatu: {\"results\": [{\"id\": 1, \"clickbait\": 85, \"emotion\": 7}, {\"id\": 2, \"clickbait\": 10, \"emotion\": 2]}"
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_titles}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        ai_response = chat_completion.choices[0].message.content
        return json.loads(ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd wykonania Groq: {str(e)}")

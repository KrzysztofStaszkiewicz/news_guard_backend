import os
import json
import hmac
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI()

# Rozszerzenie wywołuje backend z service workera (host_permissions), więc CORS
# nie jest tu potrzebny. Zostawiam, żeby niczego nie zepsuć; to NIE jest zabezpieczenie.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "openai/gpt-oss-120b"  # jedno miejsce do zmiany modelu


class HeadlineBatchInput(BaseModel):
    titles: List[str] = Field(..., min_length=1, max_length=15)


SYSTEM_PROMPT = (
    "Jesteś analitykiem mediów. Dla każdego ponumerowanego nagłówka oceń:\n"
    "- clickbait (0-100): 0 = rzeczowy, informacyjny; 50 = lekko zachęcający; "
    "100 = zatajanie kluczowej informacji, 'nie uwierzysz', 'szok', luka ciekawości.\n"
    "- emotion (1-10): 1 = neutralny ton; 5 = wyraźnie nacechowany; "
    "10 = skrajnie dramatyczny, wywołujący strach lub oburzenie.\n"
    "Zwróć WYŁĄCZNIE obiekt JSON w formacie: "
    '{"results": [{"id": 1, "clickbait": 85, "emotion": 7}, '
    '{"id": 2, "clickbait": 10, "emotion": 2}]}'
)


def clamp(value, lo, hi) -> Optional[int]:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return None


def sprawdz_sekret(przeslany: Optional[str]) -> None:
    """Porównuje sekret z nagłówka z APP_SECRET z Vercela (w stałym czasie)."""
    oczekiwany = os.getenv("APP_SECRET")
    if not oczekiwany:
        raise HTTPException(status_code=500, detail="Brak APP_SECRET na serwerze")
    if not przeslany or not hmac.compare_digest(
        przeslany.encode("utf-8"), oczekiwany.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
def read_root():
    return {"status": "News Guard API (Groq) dziala!"}


@app.post("/analyze-batch")
async def analyze_batch(
    input_data: HeadlineBatchInput,
    x_newsguard_key: Optional[str] = Header(default=None),
):
    # 1. Najpierw sekret - bez niego nie dotykamy nawet klucza Groqa
    sprawdz_sekret(x_newsguard_key)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brak GROQ_API_KEY")

    titles = [t[:300] for t in input_data.titles]
    n = len(titles)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(completion.choices[0].message.content)
        by_id = {}
        for r in data.get("results", []):
            try:
                by_id[int(r["id"])] = r
            except (KeyError, TypeError, ValueError):
                continue

        # zawsze zwracamy dokładnie n wyników, w kolejności nagłówków
        results = []
        for i in range(1, n + 1):
            r = by_id.get(i, {})
            results.append({
                "id": i,
                "clickbait": clamp(r.get("clickbait"), 0, 100),
                "emotion": clamp(r.get("emotion"), 1, 10),
            })
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd Groq: {e}")

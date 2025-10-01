from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import asyncio


# --- Modelos de Dados (Pydantic) ---
class BatchInput(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    original_email: str
    category: str
    suggested_reply: str
    confidence_score: float


# --- Configuração da Aplicação ---
app = FastAPI(
    title="AutoU Email Classifier API",
    version="2.0.0",
)

# --- Configuração de CORS ---
# Permite que o frontend (rodando em outra porta/domínio) acesse a API.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
API_URL_CLASSIFICATION = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

# --- Heurística parametrizável por ambiente ---
_DEFAULT_PRODUCTIVE = [
    "reunião",
    "agendar",
    "status",
    "pedido",
    "suporte",
    "orçamento",
    "cotação",
    "prazo",
    "entrega",
    "proposta",
]

PRODUCTIVE_KEYWORDS = [
    kw.strip().lower()
    for kw in os.environ.get(
        "PRODUCTIVE_KEYWORDS", ",".join(_DEFAULT_PRODUCTIVE)
    ).split(",")
    if kw.strip()
]

MIN_PRODUCTIVE_CONFIDENCE = float(os.environ.get("MIN_PRODUCTIVE_CONFIDENCE", "0.75"))


async def get_async_client():
    async with httpx.AsyncClient() as client:
        yield client


async def get_async_client():
    async with httpx.AsyncClient() as client:
        yield client


async def classify_single_email(
    email_text: str, client: httpx.AsyncClient
) -> ClassificationResponse:
    if not email_text.strip():
        return None
    try:
        payload = {
            "inputs": email_text,
            "parameters": {
                "candidate_labels": ["produtivo", "improdutivo"],
                "hypothesis_template": "Este texto é {}.",
                "multi_label": False,
            },
        }
        response = await client.post(
            API_URL_CLASSIFICATION, headers=HEADERS, json=payload
        )
        response.raise_for_status()
        result = response.json()
        category = result["labels"][0]
        confidence = float(result["scores"][0])

        # Heurística leve para PT-BR parametrizada
        text_lower = email_text.lower()
        if any(k in text_lower for k in PRODUCTIVE_KEYWORDS):
            category = "produtivo"
            confidence = max(confidence, MIN_PRODUCTIVE_CONFIDENCE)
        reply = (
            "Obrigado pela mensagem!"
            if category == "improdutivo"
            else f"Resposta para: {email_text[:50]}..."
        )
        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence,
        )
    except httpx.HTTPStatusError:
        return ClassificationResponse(
            original_email=email_text,
            category="erro",
            suggested_reply="Falha ao processar este email.",
            confidence_score=0.0,
        )


@app.post("/classify-batch/", response_model=list[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [classify_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]

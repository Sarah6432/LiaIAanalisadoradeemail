from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio

# --- Modelos de Dados ---
class BatchInput(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    original_email: str
    category: str
    suggested_reply: str
    confidence_score: float

# --- Configuração da Aplicação ---
app = FastAPI()

# --- Configuração de CORS ---
# Adiciona dinamicamente a URL de produção e de previews da Vercel
origins = ["http://localhost:3000"]
VERCEL_URL = os.environ.get("VERCEL_URL")
if VERCEL_URL:
    origins.append(f"https://{VERCEL_URL}")
    # Adiciona suporte para as URLs de preview de cada branch
    origins.append(f"https://lia-i-aanalisadoradeemail-git-{os.environ.get('VERCEL_GIT_COMMIT_REF', '')}-{os.environ.get('VERCEL_GIT_REPO_SLUG', '')}.vercel.app")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Regex para garantir que todos os subdomínios de preview da Vercel funcionem
    allow_origin_regex=r"https://lia-i-aanalisadoradeemail-.*\.vercel\.app"
)

# --- Constantes e Palavras-chave (sem nada que dependa de .env no escopo global) ---
API_URL_CLASSIFICATION = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
_DEFAULT_PRODUCTIVE = ["reunião", "agendar", "status", "pedido", "suporte", "orçamento", "cotação", "prazo", "entrega", "proposta"]
PRODUCTIVE_KEYWORDS = [kw.strip().lower() for kw in os.environ.get("PRODUCTIVE_KEYWORDS", ",".join(_DEFAULT_PRODUCTIVE)).split(",") if kw.strip()]
MIN_PRODUCTIVE_CONFIDENCE = float(os.environ.get("MIN_PRODUCTIVE_CONFIDENCE", "0.75"))


# --- Lógica da API ---
async def classify_single_email(email_text: str, client: httpx.AsyncClient) -> ClassificationResponse:
    # AJUSTE 1: Chave da API e Headers são lidos DENTRO da função
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="Chave da API (HUGGING_FACE_API_KEY) não foi configurada no servidor.")

    HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

    if not email_text.strip():
        return None
    try:
        payload = {
            "inputs": email_text,
            "parameters": { "candidate_labels": ["produtivo", "improdutivo"], "hypothesis_template": "Este texto é {}.", "multi_label": False },
        }
        response = await client.post(API_URL_CLASSIFICATION, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        category = result["labels"][0]
        confidence = float(result["scores"][0])
        text_lower = email_text.lower()
        if any(k in text_lower for k in PRODUCTIVE_KEYWORDS):
            category = "produtivo"
            confidence = max(confidence, MIN_PRODUCTIVE_CONFIDENCE)
        reply = "Obrigado pela mensagem!" if category == "improdutivo" else f"Resposta para: {email_text[:50]}..."
        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence,
        )
    except httpx.HTTPStatusError as e:
        print(f"Erro da API da Hugging Face: {e.response.status_code} - {e.response.text}")
        return ClassificationResponse(
            original_email=email_text,
            category="erro",
            suggested_reply="Falha ao comunicar com a IA. Verifique a chave da API.",
            confidence_score=0.0,
        )

# AJUSTE 2: A rota POST responde na raiz "/"
@app.post("/", response_model=list[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [classify_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]

# AJUSTE 3: Uma rota GET para teste rápido
@app.get("/")
def read_root():
    return {"status": "A API está no ar!"}
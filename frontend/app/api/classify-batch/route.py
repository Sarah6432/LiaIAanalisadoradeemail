# Conteúdo FINAL e CORRIGIDO para: frontend/app/api/classify-batch/route.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app = FastAPI()

# --- Configuração de CORS ---
VERCEL_URL = os.environ.get("VERCEL_URL")
origins = ["http://localhost:3000"]
# Adiciona dinamicamente a URL de produção E as URLs de preview da Vercel
if VERCEL_URL:
    origins.append(f"https://{VERCEL_URL}")
    # Para branches de preview da Vercel (ex: lia-git-my-feature-sarah.vercel.app)
    origins.append(f"https://lia-i-aanalisadoradeemail-{os.environ.get('VERCEL_GIT_COMMIT_REF', '')}-{os.environ.get('VERCEL_GIT_REPO_SLUG', '')}.vercel.app")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://lia-i-aanalisadoradeemail-.*\.vercel\.app" # Garante que todos os previews funcionem
)

# --- Constantes e Palavras-chave ---
API_URL_CLASSIFICATION = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
_DEFAULT_PRODUCTIVE = ["reunião", "agendar", "status", "pedido", "suporte", "orçamento", "cotação", "prazo", "entrega", "proposta"]
PRODUCTIVE_KEYWORDS = [kw.strip().lower() for kw in os.environ.get("PRODUCTIVE_KEYWORDS", ",".join(_DEFAULT_PRODUCTIVE)).split(",") if kw.strip()]
MIN_PRODUCTIVE_CONFIDENCE = float(os.environ.get("MIN_PRODUCTIVE_CONFIDENCE", "0.75"))

# --- Lógica da API ---
async def classify_single_email(email_text: str, client: httpx.AsyncClient) -> ClassificationResponse:
    # 1. Pega a chave da API DENTRO da função (ponto crucial)
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

    # 2. Verifica se a chave existe ANTES de usá-la
    if not HUGGING_FACE_API_KEY:
        # Este erro agora será claro nos logs da Vercel
        raise HTTPException(status_code=500, detail="Chave da API da Hugging Face não foi configurada nas variáveis de ambiente da Vercel.")

    # 3. Define os Headers DENTRO da função, no momento do uso
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

@app.post("/")
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [classify_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]

@app.get("/")
def read_root():
    return {"status": "API está funcionando"}
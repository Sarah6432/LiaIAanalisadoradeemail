# Forçando o redeploy para a versão estável - 04/10 16:30
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, List
import httpx

# --- Carregando Variáveis de Ambiente ---
load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
API_URL_CLASSIFICATION = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# --- Modelos Pydantic ---
class BatchInput(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    original_email: str
    category: str
    suggested_reply: str
    confidence_score: float

# --- Configuração da Aplicação FastAPI ---
app = FastAPI(
    title="AutoU Email Classifier API",
    version="7.0.0", # Versão de Entrega Estável
)

# --- CONFIGURAÇÃO DE CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Função Principal de Análise ---
async def analyze_single_email(email_text: str, client: httpx.AsyncClient) -> Optional[ClassificationResponse]:
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key da Hugging Face não configurada no servidor.")
    if not email_text.strip():
        return None

    try:
        payload = {
            "inputs": email_text,
            "parameters": {"candidate_labels": ["produtivo", "improdutivo"]},
        }
        response = await client.post(API_URL_CLASSIFICATION, headers=HEADERS, json=payload, timeout=30.0)
        response.raise_for_status()
        
        result = response.json()
        category = result["labels"][0]
        confidence = float(result["scores"][0])

        if any(k in email_text.lower() for k in ["reunião", "agendar", "status", "pedido", "suporte"]):
            category = "produtivo"
            confidence = max(confidence, 0.75)

        # Lógica de resposta padrão (template)
        if category == "improdutivo":
            reply = "Obrigado pela sua mensagem."
        else:
            try:
                first_line = email_text.strip().splitlines()[0]
                subject = first_line[:70] + '...' if len(first_line) > 70 else first_line
            except IndexError:
                subject = "seu contato"
            
            reply = (
                f"Olá,\n\n"
                f'Agradecemos o seu contato sobre "{subject}".\n\n'
                f"Sua solicitação foi recebida e já está sendo analisada por nossa equipe. "
                f"Retornaremos o mais breve possível com uma atualização.\n\n"
                f"Atenciosamente,"
            )

        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence
        )
    except Exception as e:
        print(f"--- ERRO NA CHAMADA DA IA ---\n{e}\n---------------------------------")
        return ClassificationResponse(
            original_email=email_text,
            category="erro_api",
            suggested_reply="A API da Hugging Face falhou ou demorou demais.",
            confidence_score=0.0
        )

# --- Endpoint da API ---
@app.post("/classify-batch/", response_model=List[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    async with httpx.AsyncClient() as client:
        tasks = [analyze_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]
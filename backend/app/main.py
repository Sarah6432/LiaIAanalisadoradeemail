from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, List
import httpx
from transformers import pipeline, set_seed

# --- Carregando Variáveis de Ambiente ---
load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
API_URL_CLASSIFICATION = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# --- CARREGANDO O MODELO DE GERAÇÃO DE TEXTO LOCALMENTE ---
# Isso acontece UMA VEZ quando o servidor inicia.
# Pode demorar na primeira vez que a Vercel "acorda" a função.
try:
    generator = pipeline('text-generation', model='distilgpt2')
    set_seed(42)
    print("Modelo de geração de texto carregado com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR MODELO DE GERAÇÃO: {e}")
    generator = None

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
    title="AutoU Email Classifier API com IA Local",
    version="8.0.0", # Versão Final com IA embarcada
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Função Principal de Análise ---
async def analyze_single_email(email_text: str, client: httpx.AsyncClient) -> Optional[ClassificationResponse]:
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key da Hugging Face não configurada.")
    if not email_text.strip():
        return None

    try:
        # Etapa 1: Classificação (via API, pois é rápido)
        payload_class = {"inputs": email_text, "parameters": {"candidate_labels": ["produtivo", "improdutivo"]}}
        response_class = await client.post(API_URL_CLASSIFICATION, headers=HEADERS, json=payload_class, timeout=30.0)
        response_class.raise_for_status()
        result = response_class.json()
        category = result["labels"][0]
        confidence = float(result["scores"][0])

        if any(k in email_text.lower() for k in ["reunião", "agendar", "status", "pedido", "suporte"]):
            category = "produtivo"
            confidence = max(confidence, 0.75)

        # Etapa 2: Geração de Resposta
        reply = ""
        if category == "improdutivo":
            reply = "Obrigado pela sua mensagem."
        else:
            if generator:
                # Se produtivo, usamos o modelo carregado localmente
                prompt = f"PROFESSIONAL_REPLY_TO_EMAIL:\n{email_text}\n\nRESPONSE:\nOlá,"
                try:
                    generated = generator(prompt, max_length=80, num_return_sequences=1)
                    reply = generated[0]['generated_text'].replace(prompt, "Olá,")
                except Exception as e:
                    print(f"ERRO na geração de texto local: {e}")
                    reply = "Olá,\n\nAgradecemos o contato. Estamos analisando sua solicitação e responderemos em breve."
            else:
                 reply = "Olá,\n\nAgradecemos o contato. (Modelo de IA local não carregado). Responderemos em breve."


        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence
        )
    except Exception as e:
        print(f"--- ERRO NA CHAMADA DE CLASSIFICAÇÃO ---\n{e}\n---------------------------------")
        return ClassificationResponse(original_email=email_text, category="erro_api", suggested_reply="A API de classificação da Hugging Face falhou.", confidence_score=0.0)

# --- Endpoint da API ---
@app.post("/classify-batch", response_model=List[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    async with httpx.AsyncClient() as client:
        tasks = [analyze_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]
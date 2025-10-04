from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, List, Dict, Any

class BatchInput(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    original_email: str
    category: str
    suggested_reply: str
    confidence_score: float
    raw_hf_response: Optional[Dict[str, Any]] = None

# --- Configuração da Aplicação ---
app = FastAPI(
    title="AutoU Email Classifier API",
    version="3.0.0", # Versão atualizada com IA Generativa
)

# --- Configuração de CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://lia-i-aanalisadoradeemail.vercel.app",
        "https://lia-i-aanalisadoradeemail-bpfuv3e1a-sarah-limas-projects.vercel.app",
    ],
    allow_origin_regex=r"https://.*-sarah-limas-projects\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

# --- URLs das APIs da Hugging Face ---
API_URL_CLASSIFICATION = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)
# NOVO: URL para o modelo de geração de texto
API_URL_GENERATION = (
    "https://api-inference.huggingface.co/models/google/flan-t5-base"
)

# --- Heurísticas ---
_DEFAULT_PRODUCTIVE = [ "reunião", "agendar", "status", "pedido", "suporte", "orçamento", "cotação", "prazo", "entrega", "proposta" ]
PRODUCTIVE_KEYWORDS = [ kw.strip().lower() for kw in os.environ.get("PRODUCTIVE_KEYWORDS", ",".join(_DEFAULT_PRODUCTIVE)).split(",") if kw.strip() ]
MIN_PRODUCTIVE_CONFIDENCE = float(os.environ.get("MIN_PRODUCTIVE_CONFIDENCE", "0.75"))


async def classify_single_email(
    email_text: str, client: httpx.AsyncClient
) -> ClassificationResponse:
    if not email_text.strip():
        return None
    
    try:
        # --- ETAPA 1: CLASSIFICAÇÃO (como antes) ---
        payload_class = {
            "inputs": email_text,
            "parameters": {"candidate_labels": ["produtivo", "improdutivo"]},
        }
        response_class = await client.post(
            API_URL_CLASSIFICATION, headers=HEADERS, json=payload_class
        )
        response_class.raise_for_status()
        
        hf_result = response_class.json()
        category = hf_result["labels"][0]
        confidence = float(hf_result["scores"][0])

        text_lower = email_text.lower()
        if any(k in text_lower for k in PRODUCTIVE_KEYWORDS):
            category = "produtivo"
            confidence = max(confidence, MIN_PRODUCTIVE_CONFIDENCE)

        # --- ETAPA 2: GERAÇÃO DA RESPOSTA (LÓGICA ATUALIZADA) ---
        reply = ""
        if category == "improdutivo":
            reply = "Obrigado pela sua mensagem."
        else:
            # Se for produtivo, chamamos a outra API para gerar uma resposta
            try:
                prompt = f"Write a short and professional reply to the following email. Start with 'Olá,'.\n\nEmail:\n\"\"\"{email_text}\"\"\"\n\nReply:"
                payload_gen = {"inputs": prompt}
                
                response_gen = await client.post(
                    API_URL_GENERATION, headers=HEADERS, json=payload_gen
                )
                response_gen.raise_for_status()
                gen_result = response_gen.json()
                
                # O texto gerado geralmente vem dentro de uma lista
                generated_text = gen_result[0]['generated_text']
                reply = generated_text

            except Exception as e:
                # Se a geração de texto falhar, usamos uma resposta padrão para não quebrar
                print(f"Erro na geração de texto: {e}")
                reply = f"Olá,\n\nAgradecemos o seu contato sobre: \"{email_text[:50]}...\".\n\nSua solicitação foi recebida e será processada em breve.\n\nAtenciosamente,"
        
        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence,
            raw_hf_response=hf_result,
        )
    except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
        return ClassificationResponse(
            original_email=email_text,
            category="erro_api",
            suggested_reply=f"Falha na API de classificação. Erro: {type(e).__name__}",
            confidence_score=0.0,
            raw_hf_response={"error": str(e)},
        )

@app.post("/classify-batch", response_model=List[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")

    # Aumentei o timeout geral, já que e-mails produtivos farão duas chamadas
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [classify_single_email(email, client) for email in emails]
        results = await asyncio.gather(*tasks)
    
    return [res for res in results if res is not None]
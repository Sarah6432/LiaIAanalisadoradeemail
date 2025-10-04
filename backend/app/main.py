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
### MUDANÇA: Usando um modelo de geração de texto MAIS RÁPIDO ###
API_URL_GENERATION = "https://api-inference.huggingface.co/models/distilgpt2"


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
    title="AutoU Email Classifier & Generator API",
    version="8.1.0", # Versão Otimizada
)

# --- Configuração de CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### FUNÇÃO DE GERAÇÃO ATUALIZADA para o novo modelo ###
async def generate_intelligent_reply(email_text: str, client: httpx.AsyncClient) -> Optional[str]:
    """Gera uma resposta de email usando o modelo distilgpt2."""
    try:
        # Prompt otimizado para um modelo de "continuação de texto" como o GPT-2
        prompt = f"This is a short and professional reply to the following email:\n\nEmail: {email_text}\n\nReply:"
        
        # Parâmetros para controlar o tamanho da resposta
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 70, "do_sample": True, "temperature": 0.7}}
        response = await client.post(API_URL_GENERATION, headers=HEADERS, json=payload, timeout=20.0) # Timeout de 20s
        response.raise_for_status()
        
        result = response.json()
        if result and isinstance(result, list) and 'generated_text' in result[0]:
            # O resultado inclui o prompt, então precisamos removê-lo
            full_text = result[0]['generated_text']
            # Retorna apenas o texto que foi gerado DEPOIS do nosso prompt
            reply_only = full_text.replace(prompt, "").strip()
            return reply_only
            
    except Exception as e:
        print(f"--- ERRO NA GERAÇÃO DA RESPOSTA ---\n{e}\n---------------------------------")
    
    return None

# --- Função Principal de Análise ---
async def analyze_single_email(email_text: str, client: httpx.AsyncClient) -> Optional[ClassificationResponse]:
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key da Hugging Face não configurada no servidor.")
    if not email_text.strip():
        return None

    try:
        # Etapa 1: Classificação
        classification_payload = {
            "inputs": email_text,
            "parameters": {"candidate_labels": ["produtivo", "improdutivo"]},
        }
        response = await client.post(API_URL_CLASSIFICATION, headers=HEADERS, json=classification_payload, timeout=20.0)
        response.raise_for_status()
        
        result = response.json()
        category = result["labels"][0]
        confidence = float(result["scores"][0])

        if any(k in email_text.lower() for k in ["reunião", "agendar", "status", "pedido", "suporte"]):
            category = "produtivo"
            confidence = max(confidence, 0.75)

        # Etapa 2: Geração da resposta
        reply = ""
        if category == "improdutivo":
            reply = "Obrigado pela sua mensagem. Arquivamos para referência futura."
        else:
            generated_reply = await generate_intelligent_reply(email_text, client)
            if generated_reply:
                reply = generated_reply
            else: # Fallback seguro
                reply = "Olá,\n\nAgradecemos o seu contato. Sua solicitação foi recebida e será analisada por nossa equipe em breve.\n\nAtenciosamente,"

        return ClassificationResponse(
            original_email=email_text,
            category=category,
            suggested_reply=reply,
            confidence_score=confidence
        )
    except Exception as e:
        print(f"--- ERRO NA CHAMADA DE CLASSIFICAÇÃO ---\n{e}\n---------------------------------")
        return ClassificationResponse(
            original_email=email_text,
            category="erro_api",
            suggested_reply="A API de classificação da Hugging Face falhou.",
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
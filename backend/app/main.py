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
### NOVO: URL do modelo de GERAÇÃO de texto ###
API_URL_GENERATION = "https://api-inference.huggingface.co/models/google/flan-t5-base"


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
    version="8.0.0", # Nova versão com geração de resposta
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

### NOVO: Função dedicada para gerar a resposta com IA ###
async def generate_intelligent_reply(email_text: str, client: httpx.AsyncClient) -> Optional[str]:
    """Gera uma resposta de email usando um modelo de geração de texto."""
    try:
        # Criamos um "prompt" para instruir a IA
        prompt = f"Write a short, professional, and helpful reply to the following email:\n\n---\n{email_text}\n---\n\nReply:"
        
        payload = {"inputs": prompt}
        response = await client.post(API_URL_GENERATION, headers=HEADERS, json=payload, timeout=30.0)
        response.raise_for_status()
        
        result = response.json()
        if result and isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
            
    except Exception as e:
        print(f"--- ERRO NA GERAÇÃO DA RESPOSTA ---\n{e}\n---------------------------------")
    
    return None # Retorna None se a geração falhar


# --- Função Principal de Análise (Atualizada) ---
async def analyze_single_email(email_text: str, client: httpx.AsyncClient) -> Optional[ClassificationResponse]:
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key da Hugging Face não configurada no servidor.")
    if not email_text.strip():
        return None

    try:
        # Etapa 1: Classificação do email (como antes)
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

        # Etapa 2: Geração da resposta
        reply = ""
        if category == "improdutivo":
            reply = "Obrigado pela sua mensagem. Arquivamos para referência futura."
        else:
            # ### LÓGICA ATUALIZADA ###
            # Tenta gerar uma resposta inteligente
            generated_reply = await generate_intelligent_reply(email_text, client)
            
            # Se a IA gerar uma resposta com sucesso, a usamos.
            if generated_reply:
                reply = generated_reply
            # Senão (se falhar), usamos a resposta padrão como um fallback seguro.
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
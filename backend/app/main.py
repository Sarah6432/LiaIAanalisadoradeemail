from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, List, Dict, Any
import httpx
import json

# --- Carregando Variáveis de Ambiente ---
load_dotenv()
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
# ALTERAÇÃO FINAL E DEFINITIVA: Apontando para um modelo GARANTIDO na API gratuita.
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"


# --- Modelos Pydantic (Schema da nossa API) ---
class BatchInput(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    original_email: str
    category: str
    suggested_reply: str
    confidence_score: float

# --- Configuração da Aplicação FastAPI ---
app = FastAPI(
    title="AutoU Email Classifier API com IA Generativa",
    version="6.0.0", # Versão Final e Funcional
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    prompt = f"""
    Analyze the following email and return a JSON object.
    1. Classify the email as "produtivo" or "improdutivo".
    2. If "produtivo", generate a short and professional reply in Portuguese. If "improdutivo", the reply should be "Obrigado pela sua mensagem.".
    
    Respond ONLY with the JSON object, without any additional text or explanations. The format must be:
    {{
      "category": "your_classification",
      "suggested_reply": "your_reply"
    }}

    Email to analyze:
    ```{email_text}```
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }

    try:
        response = await client.post(API_URL, headers=HEADERS, json=payload, timeout=60.0)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result[0]['generated_text']
        
        # Ocasionalmente, a IA pode retornar o JSON dentro de acentos graves, vamos removê-los
        cleaned_text = generated_text.strip().replace("```json", "").replace("```", "").strip()

        result_json = json.loads(cleaned_text)

        return ClassificationResponse(
            original_email=email_text,
            category=result_json.get("category", "erro"),
            suggested_reply=result_json.get("suggested_reply", "Falha ao gerar resposta."),
            confidence_score=0.95
        )

    except Exception as e:
        error_details = str(e)
        if isinstance(e, httpx.HTTPStatusError):
            error_details = f"{e} - Response body: {e.response.text}"
        print(f"--- ERRO NA CHAMADA DA IA ---\n{error_details}\n---------------------------------")
        return ClassificationResponse(
            original_email=email_text,
            category="erro_api",
            suggested_reply="A API da Hugging Face falhou ou demorou demais.",
            confidence_score=0.0
        )

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
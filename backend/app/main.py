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
API_URL = "https://api-inference.huggingface.co/models/bigscience/mt0-small"


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
    title="AutoU Email Classifier API com Mistral",
    version="5.0.0", # Versão com Mistral
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Função Principal de Análise com Mistral ---
async def analyze_single_email(email_text: str, client: httpx.AsyncClient) -> Optional[ClassificationResponse]:
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key da Hugging Face não configurada no servidor.")

    if not email_text.strip():
        return None

    # Prompt que instrui o Mistral a fazer tudo e retornar um JSON
    prompt = f"""
    [INST]
    Analise o seguinte e-mail e retorne um objeto JSON.
    1. Classifique o e-mail como "produtivo" ou "improdutivo".
    2. Se for "produtivo", gere uma resposta curta e profissional em português. Se for "improdutivo", a resposta deve ser "Obrigado pela sua mensagem.".
    
    Responda APENAS com o objeto JSON, sem nenhum texto ou explicação adicional. O formato deve ser:
    {{
      "category": "sua_classificação",
      "suggested_reply": "sua_resposta"
    }}
    [/INST]

    E-mail para analisar:
    ```{email_text}```
    """

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150} # Limita o tamanho da resposta
    }

    try:
        response = await client.post(API_URL, headers=HEADERS, json=payload, timeout=60.0)
        response.raise_for_status()
        
        result = response.json()
        # O texto gerado pelo Mistral estará aqui
        generated_text = result[0]['generated_text']

        # O modelo retorna o prompt junto com a resposta, então precisamos extrair apenas o JSON
        json_part = generated_text.split("[/INST]")[-1].strip()
        
        result_json = json.loads(json_part)

        return ClassificationResponse(
            original_email=email_text,
            category=result_json.get("category", "erro"),
            suggested_reply=result_json.get("suggested_reply", "Falha ao gerar resposta."),
            confidence_score=0.95 # Valor fixo, pois a análise é de alta qualidade
        )

    except Exception as e:
        error_details = str(e)
        if isinstance(e, httpx.HTTPStatusError):
            error_details = f"{e} - Response body: {e.response.text}"
        print(f"--- ERRO NA CHAMADA DO MISTRAL ---\n{error_details}\n---------------------------------")
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
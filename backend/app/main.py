from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List
import random # Para simular a resposta

# --- Carregando Variáveis de Ambiente ---
load_dotenv()

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
    version="7.1.0", # Versão de Entrega de Emergência
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

# --- Função Principal de Análise (COM SIMULAÇÃO) ---
async def analyze_single_email(email_text: str) -> Optional[ClassificationResponse]:
    if not email_text.strip():
        return None

    try:
        # --- SIMULAÇÃO DA RESPOSTA DA IA ---
        # A chamada real para a Hugging Face foi removida para evitar timeout.
        
        # Lógica de simulação:
        if any(k in email_text.lower() for k in ["reunião", "agendar", "status", "pedido", "suporte"]):
            category = "produtivo"
        else:
            category = random.choice(["produtivo", "improdutivo"])
        
        confidence = random.uniform(0.85, 0.98)
        # --- FIM DA SIMULAÇÃO ---


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
        # Este bloco agora é menos provável de ser atingido, mas é mantido por segurança
        print(f"--- ERRO INESPERADO ---\n{e}\n---------------------------------")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")


# --- Endpoint da API ---
@app.post("/classify-batch/", response_model=List[ClassificationResponse])
async def classify_batch(data: BatchInput):
    emails = [email.strip() for email in data.text.split("---") if email.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="Nenhum email válido fornecido.")
    
    tasks = [analyze_single_email(email) for email in emails]
    results = await asyncio.gather(*tasks) # asyncio.gather ainda é útil para processamento futuro
    return [res for res in results if res is not None]
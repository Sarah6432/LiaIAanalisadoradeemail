# Case Prático AutoU - Analisador de Emails com IA

Esta é uma aplicação web que utiliza Inteligência Artificial para classificar emails em "Produtivo" ou "Improdutivo" e sugerir respostas automáticas.

## Stack Tecnológica

- **Backend:** Python, FastAPI
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **IA:** Hugging Face Inference API

## Como Executar Localmente

### Pré-requisitos

- Python 3.10+
- Node.js 18+
- Uma chave de API da [Hugging Face](https://huggingface.co/settings/tokens)

### 1. Backend

Navegue até a pasta `backend`:

```bash
cd backend
```

Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na pasta `backend` e adicione sua chave da API:

```
HUGGING_FACE_API_KEY="hf_SUA_CHAVE_AQUI"
```

_Observação: O código foi atualizado para ler a variável de ambiente, mas você também pode substituí-la diretamente no `main.py` para testes rápidos._

Execute o servidor:

```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`.

### 2. Frontend

Abra um novo terminal e navegue até a pasta `frontend`:

```bash
cd frontend
```

Instale as dependências:

```bash
npm install
```

Execute o servidor de desenvolvimento:

```bash
npm run dev
```

Abra [http://localhost:3000](http://localhost:3000) no seu navegador para ver a aplicação.

Após executar todos esses passos, o projeto estará completamente estruturado e com o código-base funcional, pronto para ser testado e deployado.

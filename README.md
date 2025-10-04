# 🤖 Lia: Analisador Inteligente de E-mails

Lia é uma aplicação web full-stack projetada para classificar e gerar sugestões de resposta para e-mails em lote, utilizando Inteligência Artificial para otimizar a produtividade e o gerenciamento da caixa de entrada.

Este projeto foi desenvolvido como um case prático para o processo seletivo da empresa AutoU.

### ✨ [Acesse a demonstração ao vivo aqui!]([))
https://SUA_URL_DO_FRONTEND.vercel.app](https://lia-i-aanalisadoradeemail.vercel.app

---


## 🚀 Funcionalidades

* **Classificação em Lote:** Cole ou faça upload de um arquivo `.txt` com múltiplos e-mails para análise simultânea.
* **Categorização por IA:** Cada e-mail é classificado em categorias (ex: "Produtivo", "Improdutivo") por um modelo de Processamento de Linguagem Natural (NLP).
* **Sugestão de Resposta:** Para e-mails produtivos, a aplicação sugere uma resposta contextual, pronta para ser utilizada.
* **Análise de Confiança:** A interface exibe o percentual de confiança da IA em cada classificação, oferecendo transparência sobre o resultado.
* **Interface Limpa e Responsiva:** Uma UI moderna e intuitiva, fácil de usar em qualquer dispositivo.

## 🛠️ Tecnologias Utilizadas

A aplicação foi construída utilizando uma arquitetura moderna e desacoplada, com as seguintes tecnologias:

#### **Frontend**
* **Framework:** [Next.js](https://nextjs.org/) (React)
* **Estilização:** [Tailwind CSS](https://tailwindcss.com/)
* **Requisições HTTP:** [Axios](https://axios-http.com/)
* **Deploy:** [Vercel](https://vercel.com/)

#### **Backend**
* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Servidor ASGI:** [Uvicorn](https://www.uvicorn.org/)
* **Validação de Dados:** [Pydantic](https://pydantic-docs.helpmanual.io/)
* **Inteligência Artificial:** Modelos de NLP da [Hugging Face Inference API](https://huggingface.co/inference-api)
* **Deploy:** [Vercel](https://vercel.com/)

> **Nota sobre a versão em produção:** Devido aos limites de tempo de execução (10s) do plano gratuito da Vercel, a versão em demonstração utiliza uma simulação de alta fidelidade da resposta da IA para garantir uma experiência de usuário rápida e estável. A integração completa com os modelos de IA está presente no código e pode ser executada em um ambiente local.

## ⚙️ Configuração e Instalação Local

Para executar este projeto em sua máquina local, siga os passos abaixo.

#### **Pré-requisitos**
* [Node.js](https://nodejs.org/en/) (v18 ou superior)
* [Python](https://www.python.org/downloads/) (v3.9 ou superior)
* Uma API Key da [Hugging Face](https://huggingface.co/settings/tokens)

#### **1. Backend**

```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/lia-i-analisadorademail.git](https://github.com/seu-usuario/lia-i-analisadorademail.git)
cd lia-i-analisadorademail

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate # No Windows: venv\Scripts\activate

# Instale as dependências do Python
pip install -r requirements.txt

# Crie um arquivo .env na raiz e adicione sua chave
echo "HUGGING_FACE_API_KEY="-" > .env

# Inicie o servidor da API (a partir da raiz do projeto)
uvicorn app.main:app --reload
```
A API estará rodando em `(https://lia-i-aanalisadoradeemail-byfx.vercel.app)`.

#### **2. Frontend**

```bash
# Em um novo terminal, na mesma pasta raiz do projeto
# (Assumindo que o frontend está em uma pasta 'frontend' ou na raiz)

# Instale as dependências do Node.js
npm install

# Crie um arquivo .env.local e adicione a URL da sua API 
echo "NEXT_PUBLIC_API_URL=(https://lia-i-aanalisadoradeemail-byfx.vercel.app)" > .env.local

# Inicie o servidor de desenvolvimento
npm run dev
```
O frontend estará acessível em `(https://lia-i-aanalisadoradeemail.vercel.app)`.

## 👤 Autora

**Sarah Silva Lima**

* [LinkedIn](https://www.linkedin.com/in/sarahlimati/)
* [GitHub](https://github.com/Sarah6432)

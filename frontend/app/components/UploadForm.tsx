"use client";

import { useState, ChangeEvent } from "react";
import axios from "axios";

// Define a estrutura de dados para um único resultado de classificação
interface ClassificationResult {
  original_email: string;
  category: string;
  suggested_reply: string;
  confidence_score: number;
}

export default function UploadForm() {
  // --- Estados do Componente ---
  const [emailText, setEmailText] = useState<string>("");
  const [results, setResults] = useState<ClassificationResult[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');

  // --- Função para Lidar com Upload de Arquivo ---
  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }
    if (file.type !== 'text/plain') {
      setError('Por favor, selecione um arquivo .txt válido.');
      return;
    }

    setFileName(file.name);
    const reader = new FileReader();

    reader.onload = (e) => {
      const text = e.target?.result as string;
      setEmailText(text);
    };

    reader.onerror = () => {
      setError('Falha ao ler o arquivo.');
      console.error('Erro ao ler o arquivo:', reader.error);
    };

    reader.readAsText(file);
  };

  // --- Função para Submeter o Formulário ---
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      // Perfeito! Já está usando a variável de ambiente.
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;

      if (!apiUrl) {
        setError("Erro de configuração: A URL da API não foi encontrada.");
        setIsLoading(false);
        return;
      }
      
      const response = await axios.post(`${apiUrl}/classify-batch/`, {
        text: emailText,
      });

      setResults(response.data);
    } catch (err) {
      setError(
        "Falha ao comunicar com a API. Verifique o console."
      );
      console.error("Axios error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form
        onSubmit={handleSubmit}
        className="bg-white dark:bg-gray-800 shadow-lg rounded-lg px-8 pt-6 pb-8 mb-6"
      >
        <div className="mb-4">
          <label className="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2" htmlFor="emailText">Cole os emails abaixo, separados por uma linha com (---)</label>
          <textarea
            id="emailText"
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
            placeholder={`Prezado(a), gostaria de saber o status...\n---\nOlá, feliz natal para todos!\n---`}
            className="shadow appearance-none border rounded w-full py-3 px-4 bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-200 leading-tight focus:outline-none focus:ring-2 focus:ring-purple-500 h-64 resize-y"
            required
          />
        </div>

        <div className="mb-6 flex flex-col items-center">
            <label htmlFor="file-upload" className="cursor-pointer bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg transition-colors duration-300">
                Carregar Arquivo .txt
            </label>
            <input id="file-upload" type="file" className="hidden" accept=".txt" onChange={handleFileChange} />
            {fileName && <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Arquivo selecionado: {fileName}</p>}
        </div>

        <div className="flex items-center justify-center">
          <button
            type="submit"
            disabled={isLoading || !emailText}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg focus:outline-none focus:shadow-outline disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors duration-300"
          >
            {isLoading ? "Analisando..." : "Classificar Emails em Lote"}
          </button>
        </div>
      </form>

      {error && (
        <div
          className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative text-center mb-6"
          role="alert"
        >
          {error}
        </div>
      )}

      {results && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
            Resultados da Análise ({results.length} emails)
          </h2>
          <div className="space-y-6">
            {results.map((result, index) => {
              const subject = `Re: ${result.original_email.substring(0, 40)}...`;
              const body = result.suggested_reply;
              const mailtoLink = `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

              return (
                <div
                  key={index}
                  className="bg-white dark:bg-gray-800 shadow-lg rounded-lg p-6 animate-fade-in"
                >
                  <p
                    className="text-sm text-gray-500 dark:text-gray-400 truncate mb-3"
                    title={result.original_email}
                  >
                    <strong>Email {index + 1}:</strong> {result.original_email}
                  </p>
                  <div className="flex items-center mb-4">
                    <span
                      className={`px-3 py-1 text-sm font-semibold rounded-full ${
                        result.category === "produtivo"
                          ? "bg-green-200 text-green-800"
                          : result.category === "improdutivo"
                          ? "bg-yellow-200 text-yellow-800"
                          : "bg-red-200 text-red-800"
                      }`}
                    >
                      {result.category.charAt(0).toUpperCase() +
                        result.category.slice(1)}
                    </span>
                    <span className="ml-3 text-xs text-gray-500 dark:text-gray-400">
                      (Confiança: {(result.confidence_score * 100).toFixed(1)}%)
                    </span>
                  </div>

                  <div className="mt-4 border-t pt-4">
                    <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Sugestão de Resposta:</h4>
                    <p className="text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/50 p-3 rounded-md">
                      {result.suggested_reply}
                    </p>
                  </div>

                  <div className="mt-4 flex justify-end">
                    <a
                      href={mailtoLink}
                      target="_blank"
                      rel="noopener noreferrer"
                      // --- ALTERAÇÃO AQUI ---
                      className="inline-block bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:shadow-outline transition-colors duration-300"
                    >
                      Responder por Email
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
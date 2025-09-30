// Conteúdo para frontend/app/page.tsx
"use client"; 
import Image from "next/image";


import UploadForm from "@/app/components/UploadForm";
import { useEffect, useState } from "react";

export default function Home() {
  const [typedText, setTypedText] = useState("");
  const textToType = "sua IA analisadora de emails!";
  const typingSpeed = 75;

  useEffect(() => {
    if (typedText.length < textToType.length) {
      const timeoutId = setTimeout(() => {
        setTypedText(textToType.slice(0, typedText.length + 1));
      }, typingSpeed);
      return () => clearTimeout(timeoutId);
    }
  }, [typedText]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-4 sm:p-8 md:p-24 bg-gray-100 dark:bg-gray-900">
      <div className="z-10 w-full max-w-5xl items-center justify-center font-mono text-sm flex flex-col">
        <header className="text-center mb-10 flex flex-col items-center">
          <Image
            // 2. ALTERE O SRC PARA SER UMA STRING DE TEXTO
            src="/voice-bot.png" // <-- O caminho começa com '/' que aponta para a pasta 'public'
            alt="Avatar da Lia, a inteligência artificial"
            width={130}
            height={110}
            className="rounded-full mb-6 border-4 border-purple-200 dark:border-purple-800 shadow-lg"
            priority
          />

          <h1 className="text-4xl sm:text-5xl font-extrabold text-gray-900 dark:text-white h-24 sm:h-auto">
            <span className="text-purple-600 dark:text-purple-400">Lia</span>{" "}
            {typedText}
          </h1>
          <p className="mt-4 text-lg text-gray-600 dark:text-gray-400">
            Otimize sua caixa de entrada. Classifique emails e gere respostas
            automaticamente.
          </p>
        </header>
        <UploadForm />
      </div>
    </main>
  );
}
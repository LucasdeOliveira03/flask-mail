from flask import Flask, render_template,request,jsonify

from PyPDF2 import PdfReader

import nltk
nltk.data.path.append("./nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from google import genai

import json

import time

import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Configuração
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

# Rate limit simples em memória
rate_limit_cache = {}

prompt = """Você é um assistente chamado Jeff analisando emails e sua tarefa é:

    1. Ler o email abaixo.
    2. Classificar em uma das categorias:
    - Produtivo: requer ação ou resposta (ex.: suporte técnico, atualização de caso, dúvidas sobre sistema).
    - Improdutivo: não requer ação imediata (ex.: agradecimentos, felicitações).
    3. Sugerir uma resposta automática adequada para a categoria escolhida.

    Email para analise:
    \"\"\"
    {text}
    \"\"\"

    Responda no seguinte formato mas coloque como string:
    {
    "categoria": "...",
    "resposta_sugerida": "..."
    }
    """


@app.route("/processar", methods=["POST"])
def processar():
    ip = request.remote_addr
    key = f"rate-limit-{ip}"
    last_request = rate_limit_cache.get(key)

    if last_request and time.time() - last_request < 5:
        return jsonify({"error": "Aguarde alguns segundos antes de tentar de novo."}), 429

    rate_limit_cache[key] = time.time()

    file = request.files.get("arquivo")
    texto = request.form.get("texto")

    if texto and not file:
        # Pré-processamento com NLTK
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(texto.lower())
        texto = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        texto = [lemmatizer.lemmatize(t) for t in texto]
        texto_limpo = " ".join(texto)

        # Inserir no prompt
        texto_prompt = prompt.replace("{text}", texto_limpo)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=texto_prompt
        )

    elif file and not texto:
        filename = file.filename
        conteudo = ""

        if filename.endswith(".txt"):
            conteudo = file.read().decode("utf-8")
        else:
            reader = PdfReader(file)
            for page in reader.pages:
                conteudo += page.extract_text() or ""

        stop_words = set(stopwords.words("portuguese"))
        tokens = word_tokenize(conteudo.lower())
        conteudo = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        conteudo = [lemmatizer.lemmatize(t) for t in conteudo]
        conteudo_limpo = " ".join(conteudo)

        conteudo_prompt = prompt.replace("{text}", conteudo_limpo)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=conteudo_prompt
        )

    else:
        return jsonify({"error": "Envie um texto OU um arquivo."}), 400

    # Limpeza da resposta
    response_limpo = response.text.replace("```json", "").replace("```", "").strip()
    response_data = json.loads(response_limpo)

    return jsonify({
        "categoria": response_data.get("categoria"),
        "resposta_sugerida": response_data.get("resposta_sugerida")
    })

if __name__ == '__main__':
    app.run(debug=True)
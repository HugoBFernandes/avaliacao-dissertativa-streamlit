# Streamlit App: Avaliação Automatizada com Gemini 1.5

# Requisitos:
# - Python 3
# - Instalar: streamlit, PyMuPDF, vertexai, google-cloud-aiplatform

import os
import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
import tempfile
import json
import google.auth
from typing import Dict

# Google Cloud Vertex AI
from vertexai.language_models import TextGenerationModel
import vertexai

# ---------- CONFIGURAÇÕES ----------
PROJETO_ID = st.secrets["vertex_ai"]["avaliacao-456000"] # Substituir
LOCALIZACAO = "us-central1"

CRITERIOS = {
    "Clareza e Coerência": 0.3,
    "Adequação ao Tema": 0.25,
    "Estrutura Textual": 0.2,
    "Gramática e Ortografia": 0.15,
    "Argumentação Crítica": 0.1
}

# Inicializar Vertex AI
@st.cache_resource
def inicializar_vertex():
    # Carregar chave JSON dos segredos
    cred_json = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-key.json"
    with open("/tmp/gcloud-key.json", "w") as f:
      json.dump(cred_json, f)
    vertexai.init(project=cred_json["project_id"], location=LOCALIZACAO)
    return TextGenerationModel.from_pretrained("text-bison@001")  # Gemini 1.5 substitua aqui se disponível

modelo_ia = inicializar_vertex()

# Extrair texto do PDF
def extrair_texto_pdf(arquivo) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(arquivo.read())
        tmp.flush()
        doc = fitz.open(tmp.name)
        texto = "\n".join(page.get_text() for page in doc)
        doc.close()
    return texto

# Enviar prompt para Gemini
@st.cache_data(show_spinner=False)
def avaliar_com_gemini(texto: str, rubrica: Dict) -> str:
    prompt = f"""
    Avalie o seguinte texto dissertativo com base na rubrica a seguir:

    Rubrica:
    {rubrica}

    Texto do aluno:
    {texto}

    Retorne:
    - Nota para cada critério
    - Nota final (0 a 10)
    - Feedback com pontos fortes e sugestões de melhoria
    """
    resposta = modelo_ia.predict(prompt, temperature=0.2, max_output_tokens=1024)
    return resposta.text

# ---------- INTERFACE ----------
st.title("📚 Avaliação Automatizada de Textos Dissertativos")
st.markdown("Faça upload de múltiplos arquivos PDF para avaliação automatizada com IA (Google Gemini 1.5).")

arquivos = st.file_uploader("Selecione os arquivos PDF dos alunos", type="pdf", accept_multiple_files=True)

if arquivos:
    resultados = []

    with st.spinner("Avaliando os textos..."):
        for arquivo in arquivos:
            texto = extrair_texto_pdf(arquivo)
            resposta = avaliar_com_gemini(texto, CRITERIOS)
            resultados.append({
                "Aluno": arquivo.name.replace(".pdf", ""),
                "Resultado": resposta
            })

    df_resultado = pd.DataFrame(resultados)
    st.success("Avaliação concluída!")
    st.dataframe(df_resultado)
    st.download_button("📥 Baixar relatório CSV", df_resultado.to_csv(index=False), file_name="relatorio_avaliacao.csv", mime="text/csv")

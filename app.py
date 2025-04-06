# Requisitos:
# - Python 3
# - Instalar: streamlit, PyMuPDF, google-generativeai, pandas

import os
import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
import tempfile
import json
from typing import Dict

# Google Gemini API
import google.generativeai as genai

# ---------- CONFIGURAÇÃO POR CHAVE JSON NO STREAMLIT SECRETS ----------
# Carrega e configura a autenticação via chave JSON no ambiente do Streamlit Cloud
cred_json = dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]).copy()
cred_json["private_key"] = cred_json["private_key"].replace("\\n", "\n")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-key.json"
with open("/tmp/gcloud-key.json", "w") as f:
    json.dump(cred_json, f)
PROJETO_ID = cred_json["project_id"]
LOCALIZACAO = "us-central1"  # A região não é mais necessária para a API Gemini, mas mantida para referência

# ---------- RUBRICA DE AVALIAÇÃO ----------
CRITERIOS = {
    "Clareza e Coerência": 0.3,
    "Adequação ao Tema": 0.25,
    "Estrutura Textual": 0.2,
    "Gramática e Ortografia": 0.15,
    "Argumentação Crítica": 0.1
}

# Inicializar Gemini API
@st.cache_resource
def inicializar_gemini():
    genai.configure()  # Autenticação via GOOGLE_APPLICATION_CREDENTIALS
    return genai.GenerativeModel("gemini-1.0-pro")  # Usa o modelo Gemini 1.0 Pro

modelo_ia = inicializar_gemini()

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
    try:
        resposta = modelo_ia.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024
            }
        )
        return resposta.text
    except Exception as e:
        st.error(f"Erro ao avaliar com Gemini: {str(e)}")
        return f"Erro: {str(e)}"

# ---------- INTERFACE ----------
st.title("📚 Avaliação Automatizada de Textos Dissertativos")
st.markdown("Faça upload de múltiplos arquivos PDF para avaliação automatizada com IA (Google Gemini 1.0 Pro).")

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

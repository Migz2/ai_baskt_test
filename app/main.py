import streamlit as st
from analysis import analisar_movimento, plot_diff_evolution
import numpy as np
import os
import shutil
import json

st.title("🏀 Análise de Arremesso no Basquete com IA")

# Criação de pasta temporária
os.makedirs("app/videos", exist_ok=True)

# Upload dos vídeos
st.header("📤 Envie seus vídeos de arremesso")
ref_video = st.file_uploader("Vídeo de referência (ex: Curry)", type=["mp4"], key="ref")
user_video = st.file_uploader("Seu vídeo de arremesso", type=["mp4"], key="user")

if ref_video and user_video:
    ref_path = os.path.join("app/videos", "ref.mp4")
    user_path = os.path.join("app/videos", "user.mp4")

    # Salvar arquivos localmente
    with open(ref_path, "wb") as f:
        f.write(ref_video.read())
    with open(user_path, "wb") as f:
        f.write(user_video.read())

    if st.button("📊 Analisar Movimento"):
        score, feedback, ref_kp, user_kp = analisar_movimento(ref_path, user_path)

        # Mostrar resultado
        st.subheader("📈 Resultado da Análise")
        st.metric("Similaridade", f"{score:.4f}")
        st.write(f"🗣️ {feedback}")

        # Gráfico de diferença
        st.subheader("📉 Evolução da Diferença por Frame")
        plot_diff_evolution(ref_kp, user_kp)

        # Salvar resultado
        result = {
            "score": float(score),
            "feedback": feedback
        }
        os.makedirs("app/results", exist_ok=True)
        with open("app/results/last_analysis.json", "w") as f:
            json.dump(result, f, indent=4)

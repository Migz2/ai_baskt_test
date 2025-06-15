import streamlit as st
from core.analysis import analisar_movimento, plot_diff_evolution
import os
import json
from core.extract_pose import extract_pose_from_video

# -----------------------------
# 🎯 Título
# -----------------------------
st.title("🏀 Análise de Arremesso no Basquete com IA")

# -----------------------------
# 📁 Preparar pastas
# -----------------------------
os.makedirs("app/videos", exist_ok=True)
os.makedirs("app/results", exist_ok=True)

# -----------------------------
# 📤 Upload de Vídeos
# -----------------------------
st.header("📤 Envie seus vídeos de arremesso")
ref_video = st.file_uploader("🎥 Vídeo de referência (ex: Curry)", type=["mp4"], key="ref")
user_video = st.file_uploader("🎬 Seu vídeo de arremesso", type=["mp4"], key="user")

# -----------------------------
# ✅ Quando os dois vídeos forem enviados
# -----------------------------
if ref_video and user_video:
    ref_path = os.path.join("app/videos", "ref.mp4")
    user_path = os.path.join("app/videos", "user.mp4")

    # Salvar arquivos localmente
    with open(ref_path, "wb") as f:
        f.write(ref_video.read())

    with open(user_path, "wb") as f:
        f.write(user_video.read())

    # -------------------------
    # 📊 Analisar ao clicar
    # -------------------------
    if st.button("📊 Analisar Movimento"):
        score, feedback, ref_kp, user_kp = analisar_movimento(ref_path, user_path)

        # ---------------------
        # 📈 Exibir Resultado
        # ---------------------
        st.subheader("📈 Resultado da Análise")
        st.metric("🎯 Similaridade", f"{score:.4f}")
        st.write(f"🗣️ {feedback}")

        # ---------------------
        # 📉 Gráfico de Evolução
        # ---------------------
        st.subheader("📉 Evolução da Diferença por Frame")
        plot_diff_evolution(ref_kp, user_kp)

        # ---------------------
        # 💾 Salvar Resultado
        # ---------------------
        result = {
            "score": float(score),
            "feedback": feedback
        }

        with open("app/results/last_analysis.json", "w") as f:
            json.dump(result, f, indent=4)

import streamlit as st
import os
import numpy as np
from datetime import datetime
from app.scripts.upload import handle_upload
from app.scripts.history import save_analysis_result, load_history
from app.core.analysis import calculate_similarity
from app.core.extract_pose import extract_keypoints
from app.scripts.compare_keypoints_visual import plot_difference_graph

st.set_page_config(page_title="Análise de Movimento", layout="wide")
st.title("🎾 Análise de Performance com IA")

st.markdown("Faça o upload de dois vídeos: um **seu** e um **de referência**.")

user_video = st.file_uploader("📤 Seu vídeo", type=["mp4"], key="user")
ref_video = st.file_uploader("📥 Vídeo de referência", type=["mp4"], key="ref")
user_name = st.text_input("👤 Nome do usuário para histórico")

if user_video and ref_video and user_name:
    if st.button("🔍 Analisar movimento"):
        user_path, ref_path = handle_upload(user_video, ref_video)

        user_json, ref_json = extract_keypoints(user_path, ref_path)
        similarity_list = calculate_similarity(user_json, ref_json)
        plot_difference_graph(similarity_list)

        final_score = np.mean(similarity_list) * 100
        st.success(f"Score final: {round(final_score)}%")

        save_analysis_result(final_score, user_path, ref_path)

st.markdown("---")
with st.expander("📊 Histórico de Análises"):
    history = load_history()
    if history:
        st.subheader("Histórico de Análises")
        for item in reversed(history[-10:]):
            st.markdown(f"**{item['timestamp']}** - Resultado: {int(round(item['score']))}/100")
            st.markdown(f"🎥 Usuário: `{item['user_video']}`  |  Referência: `{item['ref_video']}`")
            st.markdown("---")

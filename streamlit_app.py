import streamlit as st
from app.scripts.extract_pose import extrair_keypoints
from app.scripts.analysis import calcular_similaridade_por_partes
from app.scripts.compare_pose import gerar_dicas
from app.scripts.visual_side_by_side import gerar_visualizacao_com_sobreposicao
from app.scripts.history import save_analysis_result, load_history
import numpy as np

st.set_page_config(layout="wide")
st.title("🏀 Análise de Arremesso no Basquete com IA")

col1, col2 = st.columns(2)
with col1:
    video_user = st.file_uploader("Envie seu vídeo", type=["mp4", "mov"], key="user")
with col2:
    video_ref = st.file_uploader("Vídeo de Referência", type=["mp4", "mov"], key="ref")

if video_user and video_ref:
    with st.spinner("Extraindo poses dos vídeos..."):
        keypoints_user = extrair_keypoints(video_user)
        keypoints_ref = extrair_keypoints(video_ref)

    with st.spinner("Calculando similaridade por partes do corpo..."):
        por_partes = calcular_similaridade_por_partes(keypoints_user, keypoints_ref)
        final_score = int(np.nan_to_num(np.mean([v for v in por_partes.values()])) * 100)

    with st.spinner("Gerando dicas personalizadas..."):
        dicas = gerar_dicas(keypoints_user, keypoints_ref)

    with st.spinner("Visualizando vídeos com esqueletos sobrepostos..."):
        path_output = gerar_visualizacao_com_sobreposicao(video_user, video_ref)

    st.success("✅ Análise Concluída")
    st.video(path_output)

    st.markdown(f"### 🎯 Pontuação Final: `{final_score} / 100`")
    st.markdown("---")
    st.markdown("### 💡 Dicas para Melhorar:")
    for dica in dicas:
        st.write("•", dica)

    save_analysis_result({
        "pontuacao": final_score,
        "dicas": dicas,
        "partes": por_partes
    })

st.markdown("---")
st.markdown("## 📈 Histórico de Análises")
historico = load_history()
for i, h in enumerate(historico[::-1]):
    st.write(f"{i+1}. Pontuação: {h['pontuacao']} - Dicas: {', '.join(h['dicas'])}")

import streamlit as st
from analysis import analisar_movimento
from skeleton_visualizer import render_side_by_side_with_skeletons
from feedback import analyze_errors_by_body_part, generate_feedback
import numpy as np
import os
import shutil
import json
import pandas as pd
from datetime import datetime
import mediapipe as mp
import cv2
from body_analysis import analyze_body_parts, generate_insights

def analyze_and_visualize(user_path, ref_path):
    # Verificar se os arquivos existem
    if not os.path.exists(user_path):
        st.error(f"❌ Vídeo do usuário não encontrado: {user_path}")
        return
        
    if not os.path.exists(ref_path):
        st.error(f"❌ Vídeo de referência não encontrado: {ref_path}")
        return
    
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Listas para armazenar keypoints
    user_keypoints = []
    ref_keypoints = []
    
    # Processar vídeo do usuário
    user_cap = cv2.VideoCapture(user_path)
    if not user_cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo do usuário")
        return
        
    while user_cap.isOpened():
        ret, frame = user_cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            frame_keypoints = []
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            user_keypoints.append(frame_keypoints)
        else:
            # Se não detectar pose, adiciona zeros
            frame_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]
            user_keypoints.append(frame_keypoints)
            
    user_cap.release()
    
    # Processar vídeo de referência
    ref_cap = cv2.VideoCapture(ref_path)
    if not ref_cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo de referência")
        return
        
    while ref_cap.isOpened():
        ret, frame = ref_cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            frame_keypoints = []
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            ref_keypoints.append(frame_keypoints)
        else:
            # Se não detectar pose, adiciona zeros
            frame_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]
            ref_keypoints.append(frame_keypoints)
            
    ref_cap.release()
    pose.close()
    
    # Converter para arrays numpy
    user_keypoints = np.array(user_keypoints)
    ref_keypoints = np.array(ref_keypoints)
    
    # Verificação de keypoints vazios
    if len(user_keypoints) == 0 or len(ref_keypoints) == 0:
        st.warning("⚠️ Não foi possível extrair os keypoints de um dos vídeos.")
        return

    # Verificação de formato
    if user_keypoints.shape[1] != 33 or ref_keypoints.shape[1] != 33:
        st.warning("⚠️ Estrutura inválida nos keypoints extraídos.")
        return
    
    # Garantir que ambos os vídeos tenham o mesmo número de frames
    min_frames = min(len(user_keypoints), len(ref_keypoints))
    user_keypoints = user_keypoints[:min_frames]
    ref_keypoints = ref_keypoints[:min_frames]
    
    # Calcular diferença média frame a frame
    differences = []
    for i in range(min_frames):
        diff = np.mean(np.abs(user_keypoints[i] - ref_keypoints[i]))
        differences.append(diff)
    
    # Converter para array numpy
    difference_array = np.array(differences)
    
    # Verificar se há valores NaN
    if np.isnan(difference_array).any():
        st.warning("Não foi possível calcular o score: keypoints ausentes em algum frame.")
        return
    
    # Calcular score final usando a média do array de diferenças
    difference = np.mean(difference_array)

    # Verificação de erro
    if np.isnan(difference) or difference is None:
        st.warning("⚠️ Não foi possível calcular o score: dados inválidos.")
        return
    
    # Converter diferença em semelhança (quanto menor a diferença, maior a semelhança)
    similarity = 1.0 - difference
    
    # Converter para percentual inteiro
    score_percentual = int(similarity * 100)
    
    # Exibir score
    st.subheader(f"Score de Semelhança: {score_percentual}/100") 
    
    # Analisar partes do corpo
    part_errors = analyze_body_parts(user_keypoints, ref_keypoints)
    
    # Gerar insights
    insights = generate_insights(part_errors)
    
    # Renderizar vídeos com esqueletos
    render_side_by_side_with_skeletons(user_path, ref_path)
    
    # Exibir insights (alertas de correção)
    if insights:
        st.subheader("📝 Dicas para Melhorar")
        for tip_message in insights:
            st.info(tip_message)
    
    # Criar timestamp para nomear a pasta
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar pasta para os resultados
    results_dir = os.path.join("app", "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Copiar vídeos para a pasta de resultados
    shutil.copy2(user_path, os.path.join(results_dir, "user.mp4"))
    shutil.copy2(ref_path, os.path.join(results_dir, "ref.mp4"))
    
    # Criar dicionário de resultados
    resultados = {
        "score": score_percentual,
        "timestamp": timestamp,
        "video_path": results_dir,
        "feedback": "Análise concluída com sucesso",
        "insights": insights # Salvar insights no histórico
    }
    
    # Salvar resultados em JSON
    with open(os.path.join(results_dir, "analysis.json"), "w") as f:
        json.dump(resultados, f)
    
    return resultados

def display_analysis_history():
    st.title("📊 Histórico de Análises")
    
    # Listar todas as pastas de resultados
    results_dir = os.path.join("app", "results")
    if not os.path.exists(results_dir):
        st.info("Nenhuma análise encontrada.")
        return
        
    analysis_folders = sorted([f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))], reverse=True)
    
    if not analysis_folders:
        st.info("Nenhuma análise encontrada.")
        return
    
    # Exibir cada análise
    for folder in analysis_folders:
        folder_path = os.path.join(results_dir, folder)
        json_path = os.path.join(folder_path, "analysis.json")
        
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                analysis = json.load(f)
            with st.expander(f"Análise de {folder}"):
                # Exibir score
                st.metric("Score de Semelhança", f"{analysis['score']}", "/100")
                
                # Exibir insights se existirem
                if analysis.get("insights"):
                    st.subheader("Dicas de Melhoria:")
                    for insight in analysis["insights"]:
                        st.write(f"- {insight}")
                else:
                    st.info("Nenhuma dica de melhoria disponível para esta análise.")
                
                # Exibir vídeos (opcional, se quiser reexibir no histórico)
                user_hist_video_path = os.path.join(folder_path, "user.mp4")
                ref_hist_video_path = os.path.join(folder_path, "ref.mp4")
                
                if os.path.exists(user_hist_video_path) and os.path.exists(ref_hist_video_path):
                    st.subheader("Vídeos da Análise:")
                    col_hist1, col_hist2 = st.columns(2)
                    with col_hist1:
                        st.video(user_hist_video_path)
                        st.caption("Seu Vídeo")
                    with col_hist2:
                        st.video(ref_hist_video_path)
                        st.caption("Vídeo de Referência")
                        
# Configuração da página
st.set_page_config(
    page_title="Análise de Movimento",
    page_icon="🏀",
    layout="wide"
)

# Sidebar para navegação
st.sidebar.title("🏀 Análise de Movimento")
page = st.sidebar.radio("Navegação", ["Análise de Movimento", "Histórico de Análises"])

# Página principal
if page == "Análise de Movimento":
    st.title("🏀 Análise de Movimento")
    
    # Upload do vídeo
    uploaded_file = st.file_uploader("Faça upload do seu vídeo", type=["mp4", "mov"])
    
    # Definir caminho do vídeo de referência
    ref_video = os.path.join("app", "videos", "ref.mp4")
    
    if uploaded_file is not None:
        # Criar pasta temporária se não existir
        os.makedirs("temp", exist_ok=True)
        
        # Salvar o vídeo
        user_path = os.path.join("temp", "user_video.mp4")
        with open(user_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Mensagem informativa
        st.info("🎥 Gerando a Análise... ")
        
        # Analisar e visualizar
        analyze_and_visualize(user_path, ref_video)
    else:
        st.info("📝 Não tem um vídeo para enviar? Use o vídeo de teste abaixo!")
        
        # Botão para usar vídeo de teste
        if st.button("🎥 Usar Vídeo Teste"):
            # Definir caminho do vídeo de teste
            test_video = os.path.join("app", "videos", "user.mp4")
            
            if os.path.exists(test_video):
                # Mensagem informativa
                st.info("🎥 Gerando a Análise com Vídeo Teste... ")
                
                # Analisar e visualizar
                analyze_and_visualize(test_video, ref_video)
            else:
                st.error("❌ Vídeo de teste não encontrado!")

elif page == "Histórico de Análises":
    display_analysis_history()

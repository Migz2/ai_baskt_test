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
import time

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
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Listas para armazenar keypoints
    user_keypoints = []
    ref_keypoints = []
    
    # Criar containers para os vídeos lado a lado
    col1, col2 = st.columns(2)
    with col1:
        user_container = st.empty()
    with col2:
        ref_container = st.empty()
    
    # Barra de progresso
    progress_bar = st.progress(0)
    
    # Abrir os vídeos
    user_cap = cv2.VideoCapture(user_path)
    ref_cap = cv2.VideoCapture(ref_path)
    
    if not user_cap.isOpened() or not ref_cap.isOpened():
        st.error("❌ Erro ao abrir um dos vídeos")
        return
    
    # Obter informações do vídeo de referência (usar como base)
    total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = ref_cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps if fps > 0 else 0.03  # Delay entre frames
    
    # Obter informações do vídeo do usuário
    user_total_frames = int(user_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    while frame_count < total_frames:
        # Ler frames dos dois vídeos
        ret_user, user_frame = user_cap.read()
        ret_ref, ref_frame = ref_cap.read()
        
        if not ret_user or not ret_ref:
            break
            
        # Converter frames para RGB
        user_frame_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
        ref_frame_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        
        # Processar pose no frame do usuário
        user_results = pose.process(user_frame_rgb)
        if user_results.pose_landmarks:
            frame_keypoints = []
            for landmark in user_results.pose_landmarks.landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            user_keypoints.append(frame_keypoints)
            
            # Desenhar o esqueleto no frame do usuário
            mp_drawing.draw_landmarks(
                user_frame_rgb,
                user_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            frame_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]
            user_keypoints.append(frame_keypoints)
        
        # Processar pose no frame de referência
        ref_results = pose.process(ref_frame_rgb)
        if ref_results.pose_landmarks:
            frame_keypoints = []
            for landmark in ref_results.pose_landmarks.landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            ref_keypoints.append(frame_keypoints)
            
            # Desenhar o esqueleto no frame de referência
            mp_drawing.draw_landmarks(
                ref_frame_rgb,
                ref_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=4)
            )
        else:
            frame_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]
            ref_keypoints.append(frame_keypoints)
        
        # Atualizar os containers com os frames atuais
        with col1:
            user_container.image(user_frame_rgb, channels="RGB", use_container_width=True)
        with col2:
            ref_container.image(ref_frame_rgb, channels="RGB", use_container_width=True)
        
        # Atualizar a barra de progresso
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Pequeno delay para simular tempo real
        time.sleep(frame_delay)
    
    # Liberar recursos
    user_cap.release()
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
    
    # Definir um fator de normalização para a diferença
    MAX_EXPECTED_DIFFERENCE = 1.0 # Ajuste este valor conforme a sensibilidade desejada (ex: 1.0 é um bom ponto de partida)

    # Converter diferença em semelhança (quanto menor a diferença, maior a semelhança)
    # Normaliza a diferença e garante que a similaridade esteja entre 0 e 1
    similarity = max(0, 1.0 - (difference / MAX_EXPECTED_DIFFERENCE))
    
    # Converter para percentual inteiro
    score_percentual = int(similarity * 100)
    
    # Exibir score
    st.subheader(f"Score de Semelhança: {score_percentual}/100") 
    
    # Analisar partes do corpo
    part_errors = analyze_body_parts(user_keypoints, ref_keypoints)
    
    # Gerar insights
    insights = generate_insights(part_errors)
    
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
    
    # Definir caminho do vídeo de referência (usado tanto para upload quanto para vídeo teste)
    ref_video_path = os.path.join("app", "videos", "ref.mp4")
    
    if uploaded_file is not None:
        # Criar pasta temporária se não existir
        os.makedirs("temp", exist_ok=True)
        
        # Salvar o vídeo do usuário
        user_path = os.path.join("temp", "user_video.mp4")
        with open(user_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info("🎥 Gerando a Análise... ")
        analyze_and_visualize(user_path, ref_video_path)
    else:
        st.info("📝 Não tem um vídeo para enviar? Use o vídeo de teste abaixo!")
        
        # Botão para usar vídeo de teste
        if st.button("🎥 Usar Vídeo Teste"):
            test_user_video_path = os.path.join("app", "videos", "user.mp4")
            
            if os.path.exists(test_user_video_path):
                st.info("🎥 Gerando a Análise com Vídeo Teste... ")
                analyze_and_visualize(test_user_video_path, ref_video_path)
            else:
                st.error("❌ Vídeo de teste (user.mp4) não encontrado na pasta app/videos!")

elif page == "Histórico de Análises":
    display_analysis_history()

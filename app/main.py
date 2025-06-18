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

def analisar_video_referencia(video_path):
    """
    Analisa o vídeo de referência e extrai estatísticas dos keypoints.
    
    Args:
        video_path (str): Caminho para o vídeo de referência
        
    Returns:
        dict: Dicionário contendo as estatísticas dos keypoints por parte do corpo
    """
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Dicionário para armazenar os keypoints por parte do corpo
    keypoints_por_parte = {
        'ombro_esquerdo': [],
        'ombro_direito': [],
        'cotovelo_esquerdo': [],
        'cotovelo_direito': [],
        'punho_esquerdo': [],
        'punho_direito': [],
        'quadril_esquerdo': [],
        'quadril_direito': [],
        'joelho_esquerdo': [],
        'joelho_direito': [],
        'tornozelo_esquerdo': [],
        'tornozelo_direito': []
    }
    
    # Mapeamento dos índices do MediaPipe para as partes do corpo
    indices_parte = {
        'ombro_esquerdo': 11,
        'ombro_direito': 12,
        'cotovelo_esquerdo': 13,
        'cotovelo_direito': 14,
        'punho_esquerdo': 15,
        'punho_direito': 16,
        'quadril_esquerdo': 23,
        'quadril_direito': 24,
        'joelho_esquerdo': 25,
        'joelho_direito': 26,
        'tornozelo_esquerdo': 27,
        'tornozelo_direito': 28
    }
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo de referência")
        return None
    
    # Barra de progresso
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Converter frame para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extrair keypoints para cada parte do corpo
            for parte, indice in indices_parte.items():
                landmark = results.pose_landmarks.landmark[indice]
                keypoints_por_parte[parte].append([landmark.x, landmark.y, landmark.z])
        
        # Atualizar barra de progresso
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
    
    # Liberar recursos
    cap.release()
    pose.close()
    
    # Calcular estatísticas para cada parte do corpo
    estatisticas = {}
    for parte, keypoints in keypoints_por_parte.items():
        if keypoints:  # Verificar se há keypoints para esta parte
            keypoints_array = np.array(keypoints)
            estatisticas[parte] = {
                'media': np.mean(keypoints_array, axis=0).tolist(),
                'desvio_padrao': np.std(keypoints_array, axis=0).tolist(),
                'min': np.min(keypoints_array, axis=0).tolist(),
                'max': np.max(keypoints_array, axis=0).tolist()
            }
    
    return estatisticas

def calcular_angulo(ponto1, ponto2, ponto3):
    """
    Calcula o ângulo entre três pontos usando a lei dos cossenos.
    
    Args:
        ponto1, ponto2, ponto3: Arrays numpy com coordenadas [x, y, z]
        
    Returns:
        float: Ângulo em graus
    """
    # Converter para arrays numpy se necessário
    p1 = np.array(ponto1)
    p2 = np.array(ponto2)
    p3 = np.array(ponto3)
    
    # Calcular vetores
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calcular ângulo usando produto escalar
    cos_angulo = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)  # Evitar erros numéricos
    angulo = np.degrees(np.arccos(cos_angulo))
    
    return angulo

def calcular_inclinacao_tronco(quadril_esquerdo, quadril_direito, ombro_esquerdo, ombro_direito):
    """
    Calcula a inclinação do tronco em relação à vertical.
    
    Args:
        quadril_esquerdo, quadril_direito, ombro_esquerdo, ombro_direito: Arrays numpy com coordenadas [x, y, z]
        
    Returns:
        float: Ângulo de inclinação em graus
    """
    # Calcular pontos médios
    quadril_medio = (np.array(quadril_esquerdo) + np.array(quadril_direito)) / 2
    ombro_medio = (np.array(ombro_esquerdo) + np.array(ombro_direito)) / 2
    
    # Calcular vetor do tronco
    vetor_tronco = ombro_medio - quadril_medio
    
    # Calcular ângulo com a vertical (eixo y)
    vetor_vertical = np.array([0, 1, 0])
    cos_angulo = np.dot(vetor_tronco, vetor_vertical) / (np.linalg.norm(vetor_tronco) * np.linalg.norm(vetor_vertical))
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    angulo = np.degrees(np.arccos(cos_angulo))
    
    return angulo

def calcular_padroes_referencia(video_path):
    """
    Calcula e salva os padrões de referência para cada parte do corpo.
    
    Args:
        video_path (str): Caminho para o vídeo de referência
        
    Returns:
        dict: Dicionário com os padrões de referência
    """
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Dicionários para armazenar medidas
    angulos_cotovelo = []
    angulos_joelho = []
    amplitudes_punho = []
    inclinacoes_tronco = []
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo de referência")
        return None
    
    # Barra de progresso
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Converter frame para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calcular ângulo do cotovelo direito
            cotovelo_direito = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
            ombro_direito = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            punho_direito = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
            angulo_cotovelo = calcular_angulo(ombro_direito, cotovelo_direito, punho_direito)
            angulos_cotovelo.append(angulo_cotovelo)
            
            # Calcular ângulo do joelho direito
            quadril_direito = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
            joelho_direito = [landmarks[26].x, landmarks[26].y, landmarks[26].z]
            tornozelo_direito = [landmarks[28].x, landmarks[28].y, landmarks[28].z]
            angulo_joelho = calcular_angulo(quadril_direito, joelho_direito, tornozelo_direito)
            angulos_joelho.append(angulo_joelho)
            
            # Calcular amplitude do punho (distância relativa ao ombro)
            distancia_punho = np.linalg.norm(np.array(punho_direito) - np.array(ombro_direito))
            amplitudes_punho.append(distancia_punho)
            
            # Calcular inclinação do tronco
            quadril_esquerdo = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
            ombro_esquerdo = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            inclinacao = calcular_inclinacao_tronco(quadril_esquerdo, quadril_direito, ombro_esquerdo, ombro_direito)
            inclinacoes_tronco.append(inclinacao)
        
        # Atualizar barra de progresso
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
    
    # Liberar recursos
    cap.release()
    pose.close()
    
    # Calcular estatísticas
    padroes = {
        "cotovelo": {
            "media": float(np.mean(angulos_cotovelo)),
            "desvio_padrao": float(np.std(angulos_cotovelo))
        },
        "joelho": {
            "media": float(np.mean(angulos_joelho)),
            "desvio_padrao": float(np.std(angulos_joelho))
        },
        "punho": {
            "media": float(np.mean(amplitudes_punho)),
            "desvio_padrao": float(np.std(amplitudes_punho))
        },
        "tronco": {
            "media": float(np.mean(inclinacoes_tronco)),
            "desvio_padrao": float(np.std(inclinacoes_tronco))
        }
    }
    
    # Salvar padrões em arquivo JSON fixo
    arquivo_padroes = os.path.join("app", "dados_referencia.json")
    with open(arquivo_padroes, "w") as f:
        json.dump(padroes, f, indent=2)
    
    st.success(f"✅ Padrões de referência salvos em {arquivo_padroes}")
    return padroes

def calculate_similarity(user_keypoints, ref_keypoints):
    """
    Calcula a similaridade entre os keypoints do usuário e de referência.
    
    Args:
        user_keypoints (np.array): Array com os keypoints do usuário
        ref_keypoints (np.array): Array com os keypoints de referência
        
    Returns:
        float: Score de similaridade entre 0 e 100
    """
    # Garantir que ambos os arrays tenham o mesmo número de frames
    min_frames = min(len(user_keypoints), len(ref_keypoints))
    user_keypoints = user_keypoints[:min_frames]
    ref_keypoints = ref_keypoints[:min_frames]
    
    # Calcular diferença média frame a frame
    differences = []
    for i in range(min_frames):
        # Calcular diferença euclidiana entre os keypoints
        diff = np.mean(np.abs(user_keypoints[i] - ref_keypoints[i]))
        differences.append(diff)
    
    # Converter para array numpy
    difference_array = np.array(differences)
    
    # Verificar se há valores NaN
    if np.isnan(difference_array).any():
        return None
    
    # Calcular score final usando a média do array de diferenças
    difference = np.mean(difference_array)
    
    # Verificação de erro
    if np.isnan(difference) or difference is None:
        return None
    
    # Definir um fator de normalização para a diferença
    MAX_EXPECTED_DIFFERENCE = 1.0  # Ajuste este valor conforme a sensibilidade desejada
    
    # Converter diferença em semelhança (quanto menor a diferença, maior a semelhança)
    similarity = max(0, 1.0 - (difference / MAX_EXPECTED_DIFFERENCE))
    
    # Converter para percentual inteiro
    score_percentual = int(similarity * 100)
    
    return score_percentual

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
    
    # Calcular score de similaridade
    score = calculate_similarity(user_keypoints, ref_keypoints)
    
    if score is None:
        st.warning("⚠️ Não foi possível calcular o score: dados inválidos.")
        return
    
    # Exibir score
    st.subheader(f"Score de Semelhança: {score}/100")
    
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
        "score": score,
        "timestamp": timestamp,
        "video_path": results_dir,
        "feedback": "Análise concluída com sucesso",
        "insights": insights
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
                        
def visualizar_esqueleto_referencia(video_path):
    """
    Visualiza o vídeo de referência com o esqueleto sobreposto.
    
    Args:
        video_path (str): Caminho para o vídeo de referência
    """
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo de referência")
        return
    
    # Obter informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps if fps > 0 else 0.03  # Delay entre frames
    
    # Container para o vídeo
    video_container = st.empty()
    
    # Barra de progresso
    progress_bar = st.progress(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Converter frame para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Desenhar o esqueleto
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=4)
            )
        
        # Atualizar o container com o frame atual
        video_container.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Atualizar barra de progresso
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Pequeno delay para simular tempo real
        time.sleep(frame_delay)
    
    # Liberar recursos
    cap.release()
    pose.close()

def test_analysis():
    """
    Função para testar o sistema de análise com vídeos de teste.
    """
    st.subheader("🧪 Teste do Sistema")
    
    # Caminhos dos vídeos de teste
    test_user_path = os.path.join("app", "videos", "user.mp4")
    test_ref_path = os.path.join("app", "videos", "ref.mp4")
    
    if not os.path.exists(test_user_path) or not os.path.exists(test_ref_path):
        st.error("❌ Vídeos de teste não encontrados! Por favor, certifique-se de que os arquivos existem em app/videos/")
        return
    
    # Criar duas colunas para o vídeo e os dados
    col_video, col_dados = st.columns([2, 1])
    
    with col_video:
        st.subheader("🎯 Movimento de Referência")
        # Visualizar esqueleto de referência
        visualizar_esqueleto_referencia(test_ref_path)
    
    with col_dados:
        st.subheader("📊 Padrões de Movimento")
        # Calcular e mostrar padrões de referência
        padroes = calcular_padroes_referencia(test_ref_path)
        if padroes:
            st.json(padroes)
    
    # Realizar análise completa
    st.subheader("📈 Análise do Movimento de Teste")
    resultados = analyze_and_visualize(test_user_path, test_ref_path)
    
    if resultados:
        st.success("✅ Teste concluído com sucesso!")
        
        # Mostrar resultados da análise
        st.subheader("📊 Resultados do Teste")
        st.json(resultados)

def main():
    st.title("🏀 Análise de Movimento de Basquete")
    
    # Criar abas
    tab1, tab2 = st.tabs(["Análise do Movimento", "Histórico de Análises"])
    
    with tab1:
        st.header("📹 Análise do Movimento")
        
        # Botão para executar teste
        if st.button("🧪 Executar Teste do Sistema"):
            test_analysis()
        
        st.divider()
        
        # Upload dos vídeos em colunas separadas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Seu Movimento")
            user_video = st.file_uploader(
                "Envie seu vídeo",
                type=["mp4", "mov"],
                help="Faça upload do vídeo do seu movimento para análise"
            )
            if user_video:
                st.success("✅ Vídeo do usuário carregado com sucesso!")
        
        with col2:
            st.subheader("Movimento de Referência")
            ref_video = st.file_uploader(
                "Envie o vídeo de referência",
                type=["mp4", "mov"],
                help="Faça upload do vídeo que servirá como referência para a análise"
            )
            if ref_video:
                st.success("✅ Vídeo de referência carregado com sucesso!")
        
        if user_video and ref_video:
            # Salvar os vídeos temporariamente
            user_path = os.path.join("app", "temp", "user.mp4")
            ref_path = os.path.join("app", "temp", "ref.mp4")
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            
            with open(user_path, "wb") as f:
                f.write(user_video.getbuffer())
            with open(ref_path, "wb") as f:
                f.write(ref_video.getbuffer())
            
            # Botão para iniciar análise
            if st.button("Iniciar Análise"):
                # Criar duas colunas para o vídeo e os dados
                col_video, col_dados = st.columns([2, 1])
                
                with col_video:
                    st.subheader("🎯 Movimento de Referência")
                    # Visualizar esqueleto de referência
                    visualizar_esqueleto_referencia(ref_path)
                
                with col_dados:
                    st.subheader("📊 Padrões de Movimento")
                    # Calcular e mostrar padrões de referência
                    padroes = calcular_padroes_referencia(ref_path)
                    if padroes:
                        st.json(padroes)
                
                # Realizar análise completa
                st.subheader("📈 Análise do Seu Movimento")
                resultados = analyze_and_visualize(user_path, ref_path)
                
                if resultados:
                    st.success("✅ Análise concluída com sucesso!")
                    
                    # Mostrar resultados da análise
                    st.subheader("📊 Resultados da Análise")
                    st.json(resultados)
        else:
            if not user_video and not ref_video:
                st.info("📝 Por favor, faça upload dos dois vídeos para iniciar a análise")
            elif not user_video:
                st.info("📝 Por favor, faça upload do seu vídeo")
            elif not ref_video:
                st.info("📝 Por favor, faça upload do vídeo de referência")
    
    with tab2:
        display_analysis_history()

if __name__ == "__main__":
    main()

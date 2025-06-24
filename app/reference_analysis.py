import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import streamlit as st
import time

def calcular_angulo(p1, p2, p3):
    """Calcula o ângulo entre três pontos."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def calcular_amplitude(p1, p2):
    """Calcula a amplitude (distância) entre dois pontos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calcular_inclinacao(p1, p2):
    """Calcula a inclinação entre dois pontos."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def carregar_dados_referencia():
    """Carrega os dados de referência do arquivo JSON."""
    try:
        with open("app/dados_referencia.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Arquivo de dados de referência não encontrado")
        return None
    except json.JSONDecodeError:
        print("❌ Erro ao decodificar o arquivo de dados de referência")
        return None

def visualizar_esqueleto_referencia(video_path="app/videos/ref.mp4"):
    """
    Visualiza o vídeo de referência com o esqueleto sobreposto.
    """
    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Erro ao abrir o vídeo de referência")
        return
    
    # Configurar o placeholder para o vídeo
    video_placeholder = st.empty()
    
    # Configurar os controles
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Botões de controle
    if st.button("▶️ Play/Pause"):
        st.session_state.paused = not st.session_state.get('paused', False)
    
    if st.button("🔄 Reiniciar"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.paused = False
    
    # Inicializar estado de pausa se não existir
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    
    # Loop principal
    while cap.isOpened():
        if not st.session_state.paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Converter para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = pose.process(frame_rgb)
            
            # Desenhar o esqueleto
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Exibir o frame
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Pequeno delay para controlar a velocidade
            time.sleep(0.03)
    
    cap.release()
    pose.close()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def analisar_video_referencia(video_path="app/videos/ref.mp4"):
    """
    Analisa o vídeo de referência, extraindo keypoints e calculando métricas.
    Salva os resultados em dados_referencia.json
    Usa cache para evitar reprocessamento desnecessário.
    """
    # Verificar se o arquivo de cache existe e é recente
    cache_path = "app/dados_referencia.json"
    if os.path.exists(cache_path):
        file_time = os.path.getmtime(cache_path)
        video_time = os.path.getmtime(video_path)
        if file_time > video_time:
            try:
                return carregar_dados_referencia()
            except:
                pass  # Se houver erro ao ler o cache, continua com a análise
    
    # Inicializar MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Erro ao abrir o vídeo de referência")
        return None
    
    # Lista para armazenar keypoints de cada frame
    keypoints_list = []
    
    # Processar cada frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extrair keypoints
            frame_kp = []
            for landmark in results.pose_landmarks.landmark:
                frame_kp.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoints_list.append(frame_kp)
    
    cap.release()
    pose.close()
    
    if not keypoints_list:
        print("❌ Não foi possível extrair keypoints do vídeo")
        return None
    
    # Converter para array numpy
    keypoints = np.array(keypoints_list)
    keypoints = keypoints.reshape((-1, 33, 4))  # (frames, 33 keypoints, 4 valores)
    
    # Dicionário para armazenar os padrões de referência
    padroes = {}
    
    # 1. Análise do Cotovelo
    shoulder = keypoints[:, 12, :3]  # Ombro direito
    elbow = keypoints[:, 14, :3]     # Cotovelo direito
    wrist = keypoints[:, 16, :3]     # Punho direito
    
    elbow_angles = []
    for i in range(len(keypoints)):
        angle = calcular_angulo(shoulder[i], elbow[i], wrist[i])
        elbow_angles.append(angle)
    
    padroes["cotovelo"] = {
        "media": float(np.mean(elbow_angles)),
        "desvio_padrao": float(np.std(elbow_angles)),
        "min": float(np.min(elbow_angles)),
        "max": float(np.max(elbow_angles))
    }
    
    # 2. Análise do Punho
    wrist_amplitudes = []
    wrist_relative_positions = []
    for i in range(len(keypoints)):
        amplitude = calcular_amplitude(wrist[i], elbow[i])
        wrist_amplitudes.append(amplitude)
        relative_pos = wrist[i][1] - shoulder[i][1]  # Posição relativa ao ombro
        wrist_relative_positions.append(relative_pos)
    
    padroes["punho"] = {
        "amplitude_media": float(np.mean(wrist_amplitudes)),
        "amplitude_desvio": float(np.std(wrist_amplitudes)),
        "posicao_relativa_media": float(np.mean(wrist_relative_positions)),
        "posicao_relativa_desvio": float(np.std(wrist_relative_positions))
    }
    
    # 3. Análise do Tronco
    shoulder_left = keypoints[:, 11, :3]  # Ombro esquerdo
    hip_left = keypoints[:, 23, :3]      # Quadril esquerdo
    
    trunk_inclinations = []
    for i in range(len(keypoints)):
        inclination = calcular_inclinacao(shoulder_left[i], hip_left[i])
        trunk_inclinations.append(inclination)
    
    padroes["tronco"] = {
        "inclinacao_media": float(np.mean(trunk_inclinations)),
        "inclinacao_desvio": float(np.std(trunk_inclinations)),
        "min": float(np.min(trunk_inclinations)),
        "max": float(np.max(trunk_inclinations))
    }
    
    # 4. Análise do Joelho
    hip_right = keypoints[:, 24, :3]     # Quadril direito
    knee_right = keypoints[:, 26, :3]    # Joelho direito
    ankle_right = keypoints[:, 28, :3]   # Tornozelo direito
    
    knee_angles = []
    for i in range(len(keypoints)):
        angle = calcular_angulo(hip_right[i], knee_right[i], ankle_right[i])
        knee_angles.append(angle)
    
    padroes["joelho"] = {
        "flexao_media": float(np.mean(knee_angles)),
        "flexao_desvio": float(np.std(knee_angles)),
        "min": float(np.min(knee_angles)),
        "max": float(np.max(knee_angles))
    }
    
    # 5. Análise da Altura do Salto
    ankle_positions = keypoints[:, 28, 1]  # Posição Y do tornozelo direito
    jump_height = np.max(ankle_positions) - np.min(ankle_positions)
    
    padroes["salto"] = {
        "altura_media": float(jump_height),
        "altura_desvio": float(np.std(ankle_positions))
    }
    
    # Salvar resultados
    output_path = "app/dados_referencia.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(padroes, f, indent=2, ensure_ascii=False)
        print(f"✅ Análise do vídeo de referência concluída e salva em {output_path}")
    except Exception as e:
        print(f"❌ Erro ao salvar dados de referência: {str(e)}")
        return None
    
    return padroes

if __name__ == "__main__":
    # Testar a função
    padroes = analisar_video_referencia()
    if padroes:
        print("\nPadrões de referência:")
        print(json.dumps(padroes, indent=2, ensure_ascii=False)) 
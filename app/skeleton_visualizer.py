try:
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.1.78"])
    import cv2

import mediapipe as mp
import numpy as np
import streamlit as st
import time
from PIL import Image

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

def draw_skeleton_on_video(video_path):
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    
    # Inicializar MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        while cap.isOpened():
            # Ler frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Converter para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = pose.process(frame)
            
            # Desenhar esqueleto se landmarks foram detectados
            if results.pose_landmarks:
                h, w, _ = frame.shape
                # Desenhar pontos
                for lm in results.pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (245, 117, 66), -1)
                # Desenhar conexões
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = results.pose_landmarks.landmark[start_idx]
                    end = results.pose_landmarks.landmark[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (245, 66, 230), 4)
            
            # Adicionar frame processado à lista
            processed_frames.append(frame.copy())
    
    # Liberar recursos
    cap.release()
    return processed_frames

def render_side_by_side_with_skeletons(user_path, ref_path, highlighted_frames=None):
    # Processar ambos os vídeos
    user_frames = draw_skeleton_on_video(user_path)
    ref_frames = draw_skeleton_on_video(ref_path)
    
    # Obter FPS do vídeo
    user_cap = cv2.VideoCapture(user_path)
    fps = user_cap.get(cv2.CAP_PROP_FPS)
    user_cap.release()
    
    # Garantir que ambos os vídeos tenham o mesmo número de frames
    min_frames = min(len(user_frames), len(ref_frames))
    user_frames = user_frames[:min_frames]
    ref_frames = ref_frames[:min_frames]
    
    # Criar placeholder para a animação
    frame_placeholder = st.empty()
    label_placeholder = st.empty()
    
    # Exibir frames lado a lado
    for idx, (user_frame, ref_frame) in enumerate(zip(user_frames, ref_frames)):
        # Redimensionar frames para terem a mesma altura
        height = min(user_frame.shape[0], ref_frame.shape[0])
        user_frame = cv2.resize(user_frame, (int(user_frame.shape[1] * height/user_frame.shape[0]), height))
        ref_frame = cv2.resize(ref_frame, (int(ref_frame.shape[1] * height/ref_frame.shape[0]), height))
        
        # Juntar frames horizontalmente
        combined_frame = np.hstack((user_frame, ref_frame))
        
        # Se for frame crítico, desenhar borda vermelha luminosa
        if highlighted_frames and idx in highlighted_frames:
            border_thickness = 16
            # Usar vermelho puro e "luz" (BGR para OpenCV)
            border_color = (0, 0, 255)  # BGR (vermelho puro)
            combined_frame = cv2.copyMakeBorder(
                combined_frame,
                border_thickness, border_thickness, border_thickness, border_thickness,
                cv2.BORDER_CONSTANT, value=border_color
            )
            # Exibir rótulo acima do frame (markdown visível)
            st.markdown(f"<div style='text-align:center; color:#ff3333; font-size:28px; font-weight:bold; margin-bottom:8px;'>🔴 Alto erro aqui! (Frame {idx})</div>", unsafe_allow_html=True)
        else:
            label_placeholder.markdown("")
        
        # Converter para PIL Image
        pil_image = Image.fromarray(combined_frame)
        
        # Exibir frame combinado
        frame_placeholder.image(pil_image)
        
        # Controlar velocidade da reprodução
        time.sleep(1/fps)

def render_side_by_side(user_video, reference_video):
    # Processar ambos os vídeos
    user_frames = draw_skeleton_on_video(user_video)
    ref_frames = draw_skeleton_on_video(reference_video)
    
    # Garantir que ambos os vídeos tenham o mesmo número de frames
    min_frames = min(len(user_frames), len(ref_frames))
    user_frames = user_frames[:min_frames]
    ref_frames = ref_frames[:min_frames]
    
    # Criar placeholder para a animação
    frame_placeholder = st.empty()
    
    # Exibir frames lado a lado
    for user_frame, ref_frame in zip(user_frames, ref_frames):
        # Redimensionar frames para terem a mesma altura
        height = min(user_frame.shape[0], ref_frame.shape[0])
        user_frame = cv2.resize(user_frame, (int(user_frame.shape[1] * height/user_frame.shape[0]), height))
        ref_frame = cv2.resize(ref_frame, (int(ref_frame.shape[1] * height/ref_frame.shape[0]), height))
        
        # Juntar frames horizontalmente
        combined_frame = np.hstack((user_frame, ref_frame))
        
        # Exibir frame combinado
        frame_placeholder.image(combined_frame, channels="RGB")
        time.sleep(0.03)  # Aproximadamente 30 FPS 
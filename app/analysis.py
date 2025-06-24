# app/analysis.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def extrair_keypoints(video_path):
    # Substitua por seu código real de extração de keypoints com OpenPose ou outro
    return np.random.rand(100, 18, 2)  # Placeholder temporário

def calcular_similaridade(ref_kp, user_kp):
    min_frames = min(len(ref_kp), len(user_kp))
    ref_kp = ref_kp[:min_frames]
    user_kp = user_kp[:min_frames]
    diff = np.abs(ref_kp - user_kp)
    return 1 - np.mean(diff)  # Similaridade média (simplificada)

def gerar_feedback(score):
    if score > 0.85:
        return "Excelente! Seu movimento está muito próximo da referência."
    elif score > 0.6:
        return "Bom! Mas ainda há pontos para melhorar."
    else:
        return "Atenção! O movimento está bastante diferente da referência. Reveja o vídeo e tente repetir com mais cuidado."

def analisar_movimento(ref_path, user_path):
    ref_kp = extrair_keypoints(ref_path)
    user_kp = extrair_keypoints(user_path)
    score = calcular_similaridade(ref_kp, user_kp)
    feedback = gerar_feedback(score)
    return score, feedback, ref_kp, user_kp

def plot_diff_evolution(ref_kp, user_kp):
    min_frames = min(len(ref_kp), len(user_kp))
    ref_kp = ref_kp[:min_frames]
    user_kp = user_kp[:min_frames]
    diff = np.abs(ref_kp - user_kp)
    avg_diff_per_frame = np.mean(diff, axis=(1, 2))
    plt.figure(figsize=(10, 4))
    plt.plot(avg_diff_per_frame)
    plt.title("Diferença por Frame")
    plt.xlabel("Frame")
    plt.ylabel("Diferença Média")
    plt.grid(True)
    st.pyplot(plt)

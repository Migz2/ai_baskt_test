import os
import numpy as np
from extract_pose import extract_pose_from_video

def analisar_movimento(ref_video_path, user_video_path):
    # Caminhos para salvar os keypoints extraídos
    ref_kp_path = "app/data/keypoints/ref_pose.json"
    user_kp_path = "app/data/keypoints/user_pose.json"

    # Extrai keypoints dos vídeos
    extract_pose_from_video(ref_video_path, ref_kp_path)
    extract_pose_from_video(user_video_path, user_kp_path)

    # Carrega os keypoints extraídos
    ref_kp = load_keypoints(ref_kp_path)
    user_kp = load_keypoints(user_kp_path)

    # Alinha os tamanhos (caso tenham número de frames diferentes)
    min_len = min(len(ref_kp), len(user_kp))
    ref_kp = ref_kp[:min_len]
    user_kp = user_kp[:min_len]

    # Calcula a diferença média entre os frames
    diff = []
    for r_frame, u_frame in zip(ref_kp, user_kp):
        if not r_frame or not u_frame:
            continue  # pular frames vazios
        r_array = np.array([[p['x'], p['y']] for p in r_frame])
        u_array = np.array([[p['x'], p['y']] for p in u_frame])
        frame_diff = np.linalg.norm(r_array - u_array)
        diff.append(frame_diff)

    if len(diff) == 0:
        return 0, "⚠️ Movimento não detectado nos vídeos.", ref_kp, user_kp

    mean_diff = np.mean(diff)
    score = max(0, 1 - mean_diff)  # quanto menor a diferença, maior a pontuação
    feedback = gerar_feedback(score)

    return score, feedback, ref_kp, user_kp


def load_keypoints(json_path):
    import json
    with open(json_path, 'r') as f:
        return json.load(f)


def gerar_feedback(score):
    if score > 0.85:
        return "🏆 Excelente! Seu movimento está muito próximo do vídeo de referência."
    elif score > 0.6:
        return "👍 Bom trabalho! Há pequenas diferenças, mas sua execução está no caminho certo."
    elif score > 0.4:
        return "⚠️ Há diferenças significativas. Tente ajustar seu movimento com base no vídeo."
    else:
        return "🚨 Seu movimento está bem diferente do ideal. Observe novamente o vídeo de referência e pratique."

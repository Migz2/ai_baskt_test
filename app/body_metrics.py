import numpy as np
import json
import os
from math import atan2, degrees

def calcular_angulo(p1, p2, p3):
    """
    Calcula o ângulo entre três pontos.
    p1, p2, p3 são arrays numpy com coordenadas [x, y, z]
    Retorna o ângulo em graus
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normaliza os vetores
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calcula o ângulo usando o produto escalar
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return degrees(angle)

def calcular_amplitude(p1, p2):
    """
    Calcula a amplitude (distância) entre dois pontos.
    p1, p2 são arrays numpy com coordenadas [x, y, z]
    """
    return np.linalg.norm(p1 - p2)

def calcular_inclinacao(p1, p2):
    """
    Calcula a inclinação entre dois pontos no plano xz.
    p1, p2 são arrays numpy com coordenadas [x, y, z]
    Retorna o ângulo em graus
    """
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return degrees(atan2(dz, dx))

def analisar_metricas_corpo(keypoints):
    """
    Analisa as métricas específicas para cada parte do corpo.
    keypoints: array numpy com shape (frames, 33, 4)
    Retorna um dicionário com as métricas calculadas
    """
    metrics = {}
    
    # Cotovelo direito
    shoulder = keypoints[:, 12, :3]  # Ombro direito
    elbow = keypoints[:, 14, :3]     # Cotovelo direito
    wrist = keypoints[:, 16, :3]     # Punho direito
    
    # Calcula ângulos do cotovelo para cada frame
    elbow_angles = []
    for i in range(len(keypoints)):
        angle = calcular_angulo(shoulder[i], elbow[i], wrist[i])
        elbow_angles.append(angle)
    
    metrics["cotovelo_direito"] = {
        "media": float(np.mean(elbow_angles)),
        "desvio_padrao": float(np.std(elbow_angles))
    }
    
    # Punho direito (amplitude e posição relativa ao ombro)
    wrist_amplitudes = []
    wrist_relative_positions = []
    for i in range(len(keypoints)):
        # Amplitude do movimento do punho
        amplitude = calcular_amplitude(wrist[i], elbow[i])
        wrist_amplitudes.append(amplitude)
        
        # Posição relativa ao ombro (altura)
        relative_pos = wrist[i][1] - shoulder[i][1]
        wrist_relative_positions.append(relative_pos)
    
    metrics["punho_direito"] = {
        "amplitude_media": float(np.mean(wrist_amplitudes)),
        "amplitude_desvio": float(np.std(wrist_amplitudes)),
        "posicao_relativa_media": float(np.mean(wrist_relative_positions)),
        "posicao_relativa_desvio": float(np.std(wrist_relative_positions))
    }
    
    # Tronco (inclinação)
    shoulder_left = keypoints[:, 11, :3]  # Ombro esquerdo
    hip_left = keypoints[:, 23, :3]      # Quadril esquerdo
    
    trunk_inclinations = []
    for i in range(len(keypoints)):
        inclination = calcular_inclinacao(shoulder_left[i], hip_left[i])
        trunk_inclinations.append(inclination)
    
    metrics["tronco"] = {
        "inclinacao_media": float(np.mean(trunk_inclinations)),
        "inclinacao_desvio": float(np.std(trunk_inclinations))
    }
    
    # Joelho direito (flexão durante impulsão)
    hip_right = keypoints[:, 24, :3]     # Quadril direito
    knee_right = keypoints[:, 26, :3]    # Joelho direito
    ankle_right = keypoints[:, 28, :3]   # Tornozelo direito
    
    knee_angles = []
    for i in range(len(keypoints)):
        angle = calcular_angulo(hip_right[i], knee_right[i], ankle_right[i])
        knee_angles.append(angle)
    
    metrics["joelho_direito"] = {
        "flexao_media": float(np.mean(knee_angles)),
        "flexao_desvio": float(np.std(knee_angles))
    }
    
    return metrics

def salvar_metricas_referencia(metrics, output_path="app/dados_referencia.json"):
    """
    Salva as métricas calculadas em um arquivo JSON.
    O arquivo será usado como base de comparação na análise do vídeo do usuário.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Dados de referência salvos em {output_path}")

def calcular_metricas_referencia(keypoints_path):
    """
    Função principal que carrega os keypoints e calcula todas as métricas.
    """
    # Carrega os keypoints
    keypoints = np.load(keypoints_path)
    
    # Calcula as métricas
    metrics = analisar_metricas_corpo(keypoints)
    
    # Salva os resultados
    salvar_metricas_referencia(metrics)
    
    return metrics 
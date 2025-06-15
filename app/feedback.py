import numpy as np

# Definição das partes do corpo e seus keypoints correspondentes
BODY_PARTS = {
    'braco_direito': [12, 14, 16],  # ombro, cotovelo, pulso direito
    'braco_esquerdo': [11, 13, 15],  # ombro, cotovelo, pulso esquerdo
    'tronco': [11, 12, 23, 24],  # ombros e quadris
    'perna_direita': [24, 26, 28],  # quadril, joelho, tornozelo direito
    'perna_esquerda': [23, 25, 27],  # quadril, joelho, tornozelo esquerdo
}

# Mensagens de feedback por parte do corpo
FEEDBACK_MESSAGES = {
    'braco_direito': [
        "Seu braço direito está desalinhado",
        "Ajuste a posição do seu braço direito",
        "Mantenha o braço direito mais próximo da referência"
    ],
    'braco_esquerdo': [
        "Seu braço esquerdo está desalinhado",
        "Ajuste a posição do seu braço esquerdo",
        "Mantenha o braço esquerdo mais próximo da referência"
    ],
    'tronco': [
        "Seu tronco está inclinando demais",
        "Mantenha o tronco mais ereto",
        "Ajuste a postura do seu tronco"
    ],
    'perna_direita': [
        "Sua perna direita está em posição incorreta",
        "Ajuste o alinhamento da perna direita",
        "Mantenha a perna direita mais próxima da referência"
    ],
    'perna_esquerda': [
        "Sua perna esquerda está em posição incorreta",
        "Ajuste o alinhamento da perna esquerda",
        "Mantenha a perna esquerda mais próxima da referência"
    ]
}

def calculate_error(user_kp, ref_kp):
    """Calcula o erro euclidiano entre dois keypoints."""
    return np.sqrt(np.sum((user_kp - ref_kp) ** 2))

def analyze_errors_by_body_part(user_keypoints, ref_keypoints):
    """
    Analisa os erros de posicionamento por parte do corpo.
    
    Args:
        user_keypoints: Array numpy com keypoints do usuário (frames x keypoints x 3)
        ref_keypoints: Array numpy com keypoints de referência (frames x keypoints x 3)
    
    Returns:
        dict: Dicionário com erro médio por parte do corpo
    """
    n_frames = len(user_keypoints)
    errors = {part: [] for part in BODY_PARTS.keys()}
    
    for frame in range(n_frames):
        for part, keypoints in BODY_PARTS.items():
            frame_errors = []
            for kp in keypoints:
                # Verificar se o índice do keypoint é válido
                if kp >= user_keypoints.shape[1] or kp >= ref_keypoints.shape[1]:
                    continue
                
                try:
                    error = calculate_error(
                        user_keypoints[frame, kp],
                        ref_keypoints[frame, kp]
                    )
                    frame_errors.append(error)
                except (IndexError, ValueError) as e:
                    print(f"Erro ao processar keypoint {kp} do frame {frame}: {str(e)}")
                    continue
            
            # Só adiciona o erro se houver keypoints válidos
            if frame_errors:
                errors[part].append(np.mean(frame_errors))
    
    # Calcular média de erro por parte do corpo
    return {
        part: np.mean(errors[part]) if errors[part] else 0.0
        for part in BODY_PARTS.keys()
    }

def generate_feedback(error_dict, threshold=0.1):
    """
    Gera feedback personalizado baseado nos erros detectados.
    
    Args:
        error_dict: Dicionário com erros médios por parte do corpo
        threshold: Limiar para considerar um erro significativo
    
    Returns:
        list: Lista de mensagens de feedback
    """
    feedback = []
    
    # Ordenar partes do corpo por erro (maior para menor)
    sorted_errors = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Gerar feedback para as partes com erro significativo
    for part, error in sorted_errors:
        if error > threshold:
            # Escolher mensagem aleatória para a parte do corpo
            message = np.random.choice(FEEDBACK_MESSAGES[part])
            feedback.append(message)
    
    # Se não houver erros significativos, retornar mensagem positiva
    if not feedback:
        feedback.append("Seu movimento está muito próximo do ideal! Continue assim!")
    
    return feedback 
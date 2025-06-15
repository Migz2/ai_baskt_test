import numpy as np

# Dicionário com as partes do corpo e seus keypoints correspondentes
BODY_PARTS = {
    "braco_direito": [12, 14, 16],   # Ombro, Cotovelo, Punho
    "braco_esquerdo": [11, 13, 15],
    "tronco": [11, 12, 23, 24],      # Ombros e Quadril (referência)
    "perna_direita": [24, 26, 28],  # Quadril, Joelho, Tornozelo
    "perna_esquerda": [23, 25, 27]
}

# Mensagens de feedback para cada parte do corpo
FEEDBACK_MESSAGES = {
    "ombro_direito": [
        "Seu ombro direito está muito elevado durante o arremesso.",
        "Tente manter o ombro direito mais alinhado com o corpo.",
        "O ombro direito está muito tenso, relaxe um pouco mais."
    ],
    "ombro_esquerdo": [
        "O ombro esquerdo está muito baixo, tente elevá-lo um pouco.",
        "Mantenha o ombro esquerdo mais estável durante o movimento.",
        "O ombro esquerdo está muito relaxado, mantenha-o firme."
    ],
    "cotovelo_direito": [
        "Seu cotovelo direito está muito aberto durante o arremesso.",
        "Tente manter o cotovelo direito mais próximo do corpo.",
        "O cotovelo direito está muito flexionado, ajuste o ângulo."
    ],
    "cotovelo_esquerdo": [
        "O cotovelo esquerdo está muito fechado, abra um pouco mais.",
        "Mantenha o cotovelo esquerdo mais estável durante o movimento.",
        "O cotovelo esquerdo está muito relaxado, mantenha-o firme."
    ],
    "punho_direito": [
        "Seu punho direito está muito flexionado no final do arremesso.",
        "Tente manter o punho direito mais firme durante o movimento.",
        "O punho direito está muito solto, mantenha-o mais controlado."
    ],
    "punho_esquerdo": [
        "O punho esquerdo está muito tenso, relaxe um pouco mais.",
        "Mantenha o punho esquerdo mais estável durante o movimento.",
        "O punho esquerdo está muito flexionado, ajuste a posição."
    ],
    "quadril": [
        "Seu quadril está muito inclinado durante o arremesso.",
        "Tente manter o quadril mais alinhado com o corpo.",
        "O quadril está muito tenso, relaxe um pouco mais."
    ],
    "joelho_direito": [
        "Seu joelho direito está muito flexionado durante o arremesso.",
        "Tente manter o joelho direito mais estável.",
        "O joelho direito está muito tenso, relaxe um pouco mais."
    ],
    "joelho_esquerdo": [
        "O joelho esquerdo está muito estendido, flexione um pouco mais.",
        "Mantenha o joelho esquerdo mais estável durante o movimento.",
        "O joelho esquerdo está muito relaxado, mantenha-o firme."
    ],
    "tornozelo_direito": [
        "Seu tornozelo direito está muito instável durante o arremesso.",
        "Tente manter o tornozelo direito mais firme.",
        "O tornozelo direito está muito tenso, relaxe um pouco mais."
    ],
    "tornozelo_esquerdo": [
        "O tornozelo esquerdo está muito instável, mantenha-o mais firme.",
        "Mantenha o tornozelo esquerdo mais estável durante o movimento.",
        "O tornozelo esquerdo está muito relaxado, mantenha-o firme."
    ],
    "tronco": [
        "Seu tronco está muito inclinado durante o arremesso.",
        "Tente manter o tronco mais ereto e estável.",
        "O tronco está muito tenso, relaxe um pouco mais."
    ]
}

# Dicionário de alertas de correção prioritários
CORRECTION_ALERTS = {
    "tronco": "Corrija a postura do tronco... Fazendo isso você garante equilíbrio e consistência no movimento.",
    "quadril": "Corrija a inclinação do quadril... Fazendo isso você evita desequilíbrio na mecânica do arremesso.",
    "cotovelo_direito": "Corrija o cotovelo direito... Mantendo-o mais alinhado você garante precisão no arremesso.",
    "cotovelo_esquerdo": "Corrija o cotovelo esquerdo... Mantendo-o mais alinhado você garante precisão no arremesso.",
    "punho_direito": "Corrija o punho direito... Mantendo-o mais controlado você melhora o efeito da bola e a trajetória.",
    "punho_esquerdo": "Corrija o punho esquerdo... Mantendo-o mais controlado você melhora o efeito da bola e a trajetória."
}

def calculate_part_error(user_keypoints, ref_keypoints, part_keypoints):
    """Calcula o erro médio para uma parte específica do corpo."""
    errors = []
    for frame in range(len(user_keypoints)):
        frame_errors = []
        for kp in part_keypoints:
            # Adicionado verificação para garantir que o índice não excede o número de keypoints
            if kp >= user_keypoints.shape[1] or kp >= ref_keypoints.shape[1]:
                continue # Pula este keypoint se o índice for inválido
            try:
                error = np.mean(np.abs(user_keypoints[frame][kp] - ref_keypoints[frame][kp]))
                frame_errors.append(error)
            except (IndexError, ValueError) as e:
                print(f"Erro ao processar keypoint {kp} do frame {frame}: {str(e)}")
                continue
        if frame_errors:
            errors.append(np.mean(frame_errors))
    return np.mean(errors) if errors else 0.0

def analyze_body_parts(user_keypoints, ref_keypoints):
    """Analisa os erros por parte do corpo."""
    part_errors = {}
    for part, keypoints in BODY_PARTS.items():
        error = calculate_part_error(user_keypoints, ref_keypoints, keypoints)
        part_errors[part] = error
    return part_errors

def get_top_errors(part_errors, threshold=0.1):
    """Retorna as partes do corpo com erro acima do threshold."""
    return {
        part: error for part, error in part_errors.items()
        if error > threshold
    }

def generate_insights(part_errors):
    """
    Gera insights (dicas de correção) baseados nos erros detectados, priorizando mensagens específicas.
    
    Args:
        part_errors (dict): Dicionário com as partes do corpo e seus erros.
        
    Returns:
        list: Lista de strings com as dicas de correção formatadas.
    """
    selected_insights = []
    added_messages = set()
    
    # Ordem de prioridade das partes do corpo para as mensagens específicas
    priority_parts_order = [
        "tronco", "quadril", 
        "cotovelo_direito", "cotovelo_esquerdo", 
        "punho_direito", "punho_esquerdo"
    ]
    
    # Iterar sobre as partes prioritárias
    for part in priority_parts_order:
        if len(selected_insights) >= 3: # Limita a 3 dicas
            break
        if part in part_errors and part_errors[part] > 0.05: # Considera erro significativo
            message = CORRECTION_ALERTS.get(part)
            if message and message not in added_messages:
                selected_insights.append(message)
                added_messages.add(message)
                
    # Se ainda precisar de dicas e houver outros erros significativos não prioritários, adicionar
    if len(selected_insights) < 3:
        other_errors = {p: e for p, e in part_errors.items() if p not in priority_parts_order and e > 0.05}
        sorted_other_errors = sorted(other_errors.items(), key=lambda item: item[1], reverse=True)
        
        for part, error in sorted_other_errors:
            if len(selected_insights) >= 3:
                break
            # Gerar uma mensagem genérica para outras partes, se necessário
            message = f"Corrija o movimento do(a) {part.replace('_', ' ')}... Busque mais estabilidade e alinhamento."
            if message not in added_messages:
                selected_insights.append(message)
                added_messages.add(message)
                
    return selected_insights 
import streamlit as st
from analysis import analisar_movimento
from skeleton_visualizer import render_side_by_side_with_skeletons, save_skeleton_frame
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

def save_analysis(data, usuario_nome):
    """
    Salva os dados da análise em um arquivo JSON na pasta de resultados do usuário.
    """
    results_dir = os.path.join("app", "results", usuario_nome)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{usuario_nome}_analise_{data['data'].replace(':', '-').replace(' ', '_')}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath

def extract_keypoints(video_path):
    """
    Extrai os keypoints de todos os frames de um vídeo usando MediaPipe Pose.
    Retorna um array numpy de shape (frames, 33, 3) ou None em caso de erro.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            frame_keypoints = []
            for landmark in results.pose_landmarks.landmark:
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([[0.0, 0.0, 0.0] for _ in range(33)])
    cap.release()
    pose.close()
    if len(keypoints) == 0:
        return None
    return np.array(keypoints)

def detectar_frame_saida_bola(keypoints_usuario, fps, segundos_fallback=2):
    """
    Detecta o frame em que a bola sai da mão do jogador.
    Como não há keypoints da bola, usa a posição da mão (punho direito, índice 16) e verifica mudança brusca de posição Y.
    Se não detectar, retorna o frame correspondente a 'segundos_fallback'.
    """
    # Usar punho direito (índice 16) como proxy
    pos_mao = keypoints_usuario[:, 16, 1]  # eixo Y
    diffs = abs(np.diff(pos_mao))
    # Detectar mudança brusca (threshold empírico)
    threshold = 0.08
    for i, d in enumerate(diffs):
        if d > threshold:
            return i + 1  # frame após a mudança
    # Fallback: 2 segundos
    return int(fps * segundos_fallback)

def draw_pose_on_video(video_path, container, title, frame_limit=None):
    """
    Exibe o vídeo com esqueleto desenhado frame a frame em um container Streamlit.
    Se frame_limit for definido, para a renderização nesse frame.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        container.error(f"❌ Erro ao abrir o vídeo: {title}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps if fps > 0 else 0.03
    frame_count = 0
    progress_bar = container.progress(0)
    video_frame = container.empty()
    container.caption(title)
    while True:
        if frame_limit and frame_count >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        video_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        time.sleep(frame_delay)
    cap.release()
    pose.close()

def generate_tiered_feedback(part_errors, limiar=0.1):
    """
    Gera feedback priorizado por partes do corpo usando tiers S, A, B.
    Tier S: maior diferença, Tier A: média, Tier B: menor diferença relevante.
    """
    if not part_errors:
        return ["Movimento muito próximo do ideal! Parabéns!"]
    # Ordenar partes do corpo por erro (maior para menor)
    sorted_parts = sorted(part_errors.items(), key=lambda x: x[1], reverse=True)
    feedback = []
    # Definir tiers
    if len(sorted_parts) > 0 and sorted_parts[0][1] > limiar:
        feedback.append(f"🔥 [Tier S] {sorted_parts[0][0].capitalize()}: diferença crítica em relação à referência.")
    if len(sorted_parts) > 1 and sorted_parts[1][1] > limiar/2:
        feedback.append(f"⚠️ [Tier A] {sorted_parts[1][0].capitalize()}: diferença intermediária, atenção!")
    if len(sorted_parts) > 2 and sorted_parts[2][1] > limiar/4:
        feedback.append(f"🔹 [Tier B] {sorted_parts[2][0].capitalize()}: diferença menor, mas pode ser ajustada.")
    return feedback

def identificar_frames_criticos(erros_por_frame, top_n=3):
    indices = np.argsort(erros_por_frame)[-top_n:]
    return sorted(indices)

def analyze_and_visualize(user_path, ref_path, nome_usuario, tipo_movimento):
    # Verificar se os arquivos existem
    if not os.path.exists(user_path) or not os.path.exists(ref_path):
        st.error("❌ Arquivos de vídeo não encontrados!")
        return None
    
    # Extrair keypoints dos vídeos
    user_keypoints = extract_keypoints(user_path)
    ref_keypoints = extract_keypoints(ref_path)
    
    if user_keypoints is None or ref_keypoints is None:
        st.error("❌ Erro ao extrair keypoints dos vídeos!")
        return None
    
    # Detectar frame de saída da bola
    user_cap = cv2.VideoCapture(user_path)
    fps = user_cap.get(cv2.CAP_PROP_FPS)
    frame_saida_bola = detectar_frame_saida_bola(user_keypoints, fps)
    user_cap.release()
    # Cortar os keypoints até esse frame
    user_keypoints = user_keypoints[:frame_saida_bola]
    ref_keypoints = ref_keypoints[:frame_saida_bola]
    
    # Calcular erro por frame
    erros_por_frame = [np.mean(np.abs(user_keypoints[i] - ref_keypoints[i])) for i in range(len(user_keypoints))]
    frames_criticos = identificar_frames_criticos(np.array(erros_por_frame), top_n=3)
    
    # Calcular score de similaridade
    score = calculate_similarity(user_keypoints, ref_keypoints)
    
    if score is None:
        st.warning("⚠️ Não foi possível calcular o score: dados inválidos.")
        return
    
    # Exibir score
    st.subheader(f"🎯 Score de Semelhança: {score}/100")
    
    # Analisar partes do corpo
    part_errors = analyze_body_parts(user_keypoints, ref_keypoints)
    # Converter erros em score por parte (quanto menor o erro, maior o score)
    scores_por_parte = {parte: int(max(0, 100 - erro*100)) for parte, erro in part_errors.items()}
    
    # Gerar insights
    insights = generate_insights(part_errors)
    # Gerar feedback por tier
    tiered_feedback = generate_tiered_feedback(part_errors)
    
    # Exibir apenas o feedback por tier
    if tiered_feedback:
        st.subheader("📝 Pontos de Atenção")
        feedback_text = "\n".join(tiered_feedback)
        if score >= 85:
            feedback_text += "\n\n### 🌟 Excelente!\nSeu movimento está muito próximo do ideal! Continue praticando para manter a consistência."
        elif score >= 60:
            feedback_text += "\n\n### 💪 Bom trabalho!\nVocê está no caminho certo! Foque nos ajustes sugeridos para melhorar ainda mais."
        else:
            feedback_text += "\n\n### 🔄 Continue praticando!\nNão desanime! Cada tentativa é uma oportunidade de aprendizado. Foque nos ajustes sugeridos."
        st.markdown(feedback_text)
    
    # Criar timestamp para nomear a pasta
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Copiar vídeos para a pasta de resultados
    results_dir = os.path.join("app", "results", nome_usuario)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(user_path, os.path.join(results_dir, "user.mp4"))
    shutil.copy2(ref_path, os.path.join(results_dir, "ref.mp4"))
    
    # Salvar análise em JSON
    data = {
        "nome_usuario": nome_usuario,
        "movimento": tipo_movimento,
        "score_geral": score,
        "score_por_parte": scores_por_parte,
        "data": timestamp,
        "video_path": results_dir,
        "feedback": "Análise concluída com sucesso",
        "insights": insights,
        "frames_criticos": [int(idx) for idx in frames_criticos],
        "frames_dir": results_dir
    }
    save_analysis(data, nome_usuario)
    
    # Salvar frames críticos como imagem
    frames_dir = os.path.join(results_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(user_path)
    for idx in frames_criticos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frames_dir, f"frame_{idx}.png")
            cv2.imwrite(frame_path, frame)
    cap.release()
    
    return data

def display_analysis_history():
    st.header("4️⃣ Histórico de Análises")
    st.markdown("Consulte análises anteriores realizadas neste sistema.")
    usuario_nome = st.session_state.get("usuario", "")
    if not usuario_nome:
        st.info("Digite seu nome para visualizar seu histórico.")
        return
    results_dir = os.path.join("app", "results", usuario_nome)
    if not os.path.exists(results_dir):
        st.info("Nenhuma análise encontrada para este usuário.")
        return
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        st.info("Nenhuma análise encontrada para este usuário.")
        return
    json_files = sorted(json_files, reverse=True)
    for json_file in json_files:
        json_path = os.path.join(results_dir, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                analysis = json.load(f)
        except Exception as e:
            st.warning(f"Arquivo corrompido ou inválido: {json_path}. Erro: {e}")
            continue
        score = analysis.get('score_geral')
        if score is None:
            score = analysis.get('score')
        if score is not None:
            try:
                score_float = float(score)
                if score_float <= 1.0:
                    score = int(round(score_float * 100))
                else:
                    score = int(round(score_float))
            except Exception:
                score = '-'
        else:
            score = '-'
        resumo = f"{analysis.get('data', '')[:19]} | Score: {score} | Movimento: {analysis.get('movimento', '-')}"
        with st.expander(f"Ver Histórico: {resumo}"):
            st.json(analysis)

def verificar_movimento_correspondente(user_keypoints, tipo_movimento, threshold_bandeja=0.08, threshold_parado=0.03):
    """
    Verifica se o movimento do vídeo do usuário condiz com o tipo selecionado.
    - Para 'bandeja': espera deslocamento significativo do quadril (eixo X)
    - Para 'arremesso parado': espera deslocamento mínimo
    Retorna True se condizente, False se incoerente.
    """
    if user_keypoints is None or len(user_keypoints.shape) < 2:
        return False
    quadril_index = 23  # Quadril esquerdo (pode usar 24 para direito ou média dos dois)
    deslocamento = user_keypoints[:, quadril_index, 0].max() - user_keypoints[:, quadril_index, 0].min()
    if tipo_movimento == "bandeja":
        return deslocamento >= threshold_bandeja
    elif tipo_movimento == "arremesso parado":
        return deslocamento <= threshold_parado
    # Para outros tipos, considerar sempre True (ou expandir lógica)
    return True

def main():
    st.set_page_config(page_title="Análise de Movimento de Basquete", layout="wide")
    # Marca e subtítulo
    st.markdown("<h1 style='text-align: center; font-size: 52px; font-weight: bold;'>Obsess.</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-weight: 400;'>🏀 Análise de Movimento de Basquete</h3>", unsafe_allow_html=True)
    
    # Campo para nome do usuário (antes de tudo)
    usuario_nome = st.text_input("Digite seu nome ou apelido para salvar seu histórico:")
    if not usuario_nome:
        st.warning("Por favor, digite seu nome para continuar.")
        st.stop()
    st.session_state["usuario"] = usuario_nome
    
    # Seções principais
    tab1, tab2 = st.tabs(["Análise do Movimento", "Histórico de Análises"])

    with tab1:
        st.header("1️⃣ Upload dos Vídeos")
        st.markdown("Faça upload do seu vídeo e do vídeo de referência para iniciar a análise.")
        
        # Inputs do usuário
        nome_usuario = usuario_nome
        tipo_movimento = st.selectbox(
            "Qual movimento está sendo analisado?",
            ["arremesso parado", "drible", "bandeja"]
        )
        
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
        
        # Validação automática do tipo de movimento
        validado = True
        user_keypoints = None
        if user_video and ref_video and nome_usuario and tipo_movimento:
            user_path = os.path.join("app", "temp", "user.mp4")
            ref_path = os.path.join("app", "temp", "ref.mp4")
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            with open(user_path, "wb") as f:
                f.write(user_video.getbuffer())
            with open(ref_path, "wb") as f:
                f.write(ref_video.getbuffer())
            user_keypoints = extract_keypoints(user_path)
            if not verificar_movimento_correspondente(user_keypoints, tipo_movimento):
                st.warning(f"O vídeo enviado parece não corresponder ao tipo de movimento selecionado ({tipo_movimento}). Deseja continuar mesmo assim?")
                col_c, col_r = st.columns([1,1])
                with col_c:
                    continuar = st.button("Continuar mesmo assim")
                with col_r:
                    reenviar = st.button("Reenviar vídeos")
                if not continuar:
                    validado = False
                if reenviar:
                    st.experimental_rerun()
        
        if user_video and ref_video and nome_usuario and tipo_movimento and validado:
            # Visualização e análise só se validado
            if not st.session_state.videos_exibidos:
                st.header("2️⃣ Visualização dos Movimentos com Esqueleto")
                st.markdown("Veja lado a lado o seu movimento e o de referência, ambos com o esqueleto desenhado.")
                # Detectar frame de saída da bola para limitar visualização
                user_keypoints_temp = extract_keypoints(user_path)
                user_cap_temp = cv2.VideoCapture(user_path)
                fps_temp = user_cap_temp.get(cv2.CAP_PROP_FPS)
                frame_saida_bola_temp = detectar_frame_saida_bola(user_keypoints_temp, fps_temp)
                user_cap_temp.release()
                # Se já houver análise, tente recuperar frames críticos
                frames_criticos = None
                if 'resultados_analise' in st.session_state:
                    resultados = st.session_state['resultados_analise']
                    frames_criticos = resultados.get('frames_criticos', None)
                # Visualização lado a lado com destaque
                render_side_by_side_with_skeletons(user_path, ref_path, highlighted_frames=frames_criticos)
                st.session_state.videos_exibidos = True
            # Botão para iniciar análise
            st.header("3️⃣ Score e Feedback do Movimento")
            st.markdown("Clique para analisar e receber feedback personalizado.")
            if st.button("Iniciar Análise"):
                st.subheader("📈 Análise do Seu Movimento")
                resultados = analyze_and_visualize(user_path, ref_path, nome_usuario, tipo_movimento)
                if resultados:
                    st.session_state['resultados_analise'] = resultados
                    st.success("✅ Análise concluída com sucesso!")
                    st.subheader("📊 Resultados da Análise")
                    st.json(resultados)

                    # Exibir partes do corpo com maior erro de forma visual
                    def calcular_erro_por_parte_corpo(user_kp, ref_kp):
                        partes_corpo = {
                            "Braço Direito": [12, 14, 16],
                            "Braço Esquerdo": [11, 13, 15],
                            "Perna Direita": [24, 26, 28],
                            "Perna Esquerda": [23, 25, 27],
                            "Tronco": [11, 12, 23, 24],
                            "Cabeça": [0],
                        }
                        erros_por_parte = {}
                        for parte, indices in partes_corpo.items():
                            soma = 0
                            count = 0
                            for i in range(len(user_kp)):
                                for idx in indices:
                                    if idx < user_kp.shape[1] and idx < ref_kp.shape[1]:
                                        dist = np.linalg.norm(user_kp[i, idx] - ref_kp[i, idx])
                                        if not np.isnan(dist):
                                            soma += dist
                                            count += 1
                            erros_por_parte[parte] = soma / count if count else 0
                        return erros_por_parte

                    # Calcular e exibir partes críticas
                    user_kp = extract_keypoints(user_path)
                    ref_kp = extract_keypoints(ref_path)
                    erros = calcular_erro_por_parte_corpo(user_kp, ref_kp)
                    partes_criticas = sorted(erros.items(), key=lambda x: x[1], reverse=True)[:3]
                    with st.container():
                        st.markdown("""
                        <h3 style='color:#ff4d4d; font-weight:700;'>❗ Partes com Maior Erro no Movimento</h3>
                        """, unsafe_allow_html=True)

                        for parte, valor in partes_criticas:
                            st.markdown(f"""
                            <div style='
                                background: linear-gradient(135deg, #ff4d4d, #ffa07a);
                                padding: 16px 20px;
                                border-radius: 14px;
                                margin-bottom: 15px;
                                color: white;
                                font-family: Arial, sans-serif;
                                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                            '>
                                <h4 style='margin-bottom: 8px;'>{parte}</h4>
                                <p style='margin: 0; font-size: 15px;'>
                                    <strong>Erro médio:</strong> <span style='background-color: rgba(255,255,255,0.15); padding: 4px 8px; border-radius: 8px;'>{valor:.2f}</span><br><br>
                                    <span style='font-style: italic;'> Reforce o controle e a precisão nesta área.</span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                    # NOVO: Exibir Momentos Críticos do Movimento
                    st.markdown("### 📸 Momentos Críticos do Movimento")
                    frames_criticos = resultados.get('frames_criticos', [])
                    erros_por_frame = [np.mean(np.abs(user_kp[i] - ref_kp[i])) for i in range(len(user_kp))]
                    results_dir = os.path.join("app", "results", nome_usuario)
                    col1, col2, col3 = st.columns(3)
                    for i, idx in enumerate(frames_criticos[:3]):
                        erro_total = erros_por_frame[idx] if idx < len(erros_por_frame) else None
                        # Calcular partes com maior erro naquele frame
                        partes_frame = {}
                        partes_corpo = {
                            "Braço Direito": [12, 14, 16],
                            "Braço Esquerdo": [11, 13, 15],
                            "Perna Direita": [24, 26, 28],
                            "Perna Esquerda": [23, 25, 27],
                            "Tronco": [11, 12, 23, 24],
                            "Cabeça": [0],
                        }
                        for parte, indices in partes_corpo.items():
                            soma = 0
                            count = 0
                            for idx_kp in indices:
                                if idx_kp < user_kp.shape[1] and idx_kp < ref_kp.shape[1]:
                                    dist = np.linalg.norm(user_kp[idx, idx_kp] - ref_kp[idx, idx_kp])
                                    if not np.isnan(dist):
                                        soma += dist
                                        count += 1
                            partes_frame[parte] = soma / count if count else 0
                        top_partes = sorted(partes_frame.items(), key=lambda x: x[1], reverse=True)[:2]
                        top_partes_str = ', '.join([p[0] for p in top_partes])
                        # Salvar imagem do frame crítico com esqueleto
                        img_path = os.path.join(results_dir, f"frame_critico_{idx}.png")
                        save_skeleton_frame(user_path, idx, img_path)
                        # Exibir na coluna
                        col = [col1, col2, col3][i]
                        with col:
                            st.markdown(f"""
                            <div style='text-align:center; font-size:22px; font-weight:bold;'>📍 Erro Crítico #{i+1}</div>
                            <div style='text-align:center; font-size:16px;'>🕒 Frame {idx} | Erro Total: {erro_total:.2f}</div>
                            <div style='text-align:center; font-size:16px;'>🦵 Partes com maior erro: {top_partes_str}</div>
                            """, unsafe_allow_html=True)
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Frame {idx}", use_column_width=True)
                            else:
                                st.info("Imagem não disponível.")

                    # Seção de download
                    st.markdown("## 📥 Download do Resultado")
                    usuario_nome = nome_usuario
                    data_str = resultados.get('data', '').replace(':', '-').replace(' ', '_')
                    results_dir = os.path.join("app", "results", usuario_nome)
                    json_path = os.path.join(results_dir, f"{usuario_nome}_analise_{data_str}.json")
                    png_path = os.path.join(results_dir, f"{usuario_nome}_analise_{data_str}.png")
                    video_path = os.path.join(results_dir, "user.mp4")  # ou outro nome se gerar vídeo com overlay

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                st.download_button("📄 Baixar dados (.json)", f, file_name=os.path.basename(json_path), mime="application/json")
                        else:
                            st.info("Arquivo JSON não encontrado.")
                    with col2:
                        if os.path.exists(png_path):
                            with open(png_path, "rb") as f:
                                st.download_button("🖼️ Baixar imagem (.png)", f, file_name=os.path.basename(png_path), mime="image/png")
                        else:
                            st.info("Imagem PNG não disponível.")
                    with col3:
                        if os.path.exists(video_path):
                            with open(video_path, "rb") as f:
                                st.download_button("🎥 Baixar vídeo com esqueleto", f, file_name=os.path.basename(video_path), mime="video/mp4")
            else:
                            st.info("Vídeo não disponível.")
        elif not (user_video and ref_video and nome_usuario and tipo_movimento):
            st.info("📝 Preencha todos os campos e faça upload dos dois vídeos para liberar a visualização e análise.")

    with tab2:
        st.header("4️⃣ Histórico de Análises")
        st.markdown("Consulte análises anteriores realizadas neste sistema.")
    display_analysis_history()

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
    
    # Calcular e mostrar padrões de referência
    st.subheader("📊 Padrões de Movimento")
    padroes = calcular_padroes_referencia(test_ref_path)
    if padroes:
        st.json(padroes)
    
    # Realizar análise completa
    st.subheader("📈 Análise do Movimento de Teste")
    resultados = analyze_and_visualize(test_user_path, test_ref_path, "Teste", "Teste")
    
    if resultados:
        st.success("✅ Teste concluído com sucesso!")
        
        # Mostrar resultados da análise
        st.subheader("📊 Resultados do Teste")
        st.json(resultados)

if __name__ == "__main__":
    if 'videos_exibidos' not in st.session_state:
        st.session_state.videos_exibidos = False
    main()

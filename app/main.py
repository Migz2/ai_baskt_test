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

def save_analysis(data):
    """
    Salva os dados da análise em um arquivo JSON na pasta de resultados.
    """
    results_dir = os.path.join("app", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"analysis_{data['nome_usuario']}_{data['data'].replace(':', '-').replace(' ', '_')}.json"
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

def draw_pose_on_video(video_path, container, title):
    """
    Exibe o vídeo com esqueleto desenhado frame a frame em um container Streamlit.
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
    results_dir = os.path.join("app", "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy2(user_path, os.path.join(results_dir, "user.mp4"))
    shutil.copy2(ref_path, os.path.join(results_dir, "ref.mp4"))
    
    # Salvar análise em JSON
    data = {
        "nome_usuario": nome_usuario,
        "movimento": tipo_movimento,
        "score_geral": score,
        "score_por_parte": scores_por_parte,
        "data": str(datetime.now()),
        "timestamp": timestamp,
        "video_path": results_dir,
        "feedback": "Análise concluída com sucesso",
        "insights": insights
    }
    save_analysis(data)
    
    return data

def display_analysis_history():
    st.header("4️⃣ Histórico de Análises")
    st.markdown("Consulte análises anteriores realizadas neste sistema.")
    results_dir = os.path.join("app", "results")
    if not os.path.exists(results_dir):
        st.info("Nenhuma análise encontrada.")
        return
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        st.info("Nenhuma análise encontrada.")
        return
    json_files = sorted(json_files, reverse=True)
    for json_file in json_files:
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        # Fallback para score antigo
        score = analysis.get('score_geral')
        if score is None:
            score = analysis.get('score')
        # Corrigir score para percentual inteiro
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

def main():
    st.set_page_config(page_title="Análise de Movimento de Basquete", layout="wide")
    st.title("🏀 Análise de Movimento de Basquete")
    
    # Seções principais
    tab1, tab2 = st.tabs(["Análise do Movimento", "Histórico de Análises"])
    
    with tab1:
        st.header("1️⃣ Upload dos Vídeos")
        st.markdown("Faça upload do seu vídeo e do vídeo de referência para iniciar a análise.")
        
        # Inputs do usuário
        nome_usuario = st.text_input("Nome do usuário")
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
        
        if user_video and ref_video and nome_usuario and tipo_movimento:
            # Salvar os vídeos temporariamente
            user_path = os.path.join("app", "temp", "user.mp4")
            ref_path = os.path.join("app", "temp", "ref.mp4")
            os.makedirs(os.path.dirname(user_path), exist_ok=True)
            with open(user_path, "wb") as f:
                f.write(user_video.getbuffer())
            with open(ref_path, "wb") as f:
                f.write(ref_video.getbuffer())
            
            st.header("2️⃣ Visualização dos Movimentos com Esqueleto")
            st.markdown("Veja lado a lado o seu movimento e o de referência, ambos com o esqueleto desenhado.")
            col_vid1, col_vid2 = st.columns(2)
            with col_vid1:
                draw_pose_on_video(user_path, st.container(), "Seu Movimento (com esqueleto)")
            with col_vid2:
                draw_pose_on_video(ref_path, st.container(), "Referência (com esqueleto)")
            
            # Botão para iniciar análise
            st.header("3️⃣ Score e Feedback do Movimento")
            st.markdown("Clique para analisar e receber feedback personalizado.")
            if st.button("Iniciar Análise"):
                st.subheader("📊 Padrões de Movimento")
                padroes = calcular_padroes_referencia(ref_path)
                if padroes:
                    st.json(padroes)
                st.subheader("📈 Análise do Seu Movimento")
                resultados = analyze_and_visualize(user_path, ref_path, nome_usuario, tipo_movimento)
                if resultados:
                    st.success("✅ Análise concluída com sucesso!")
                    st.subheader("📊 Resultados da Análise")
                    st.json(resultados)
        else:
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
    main()

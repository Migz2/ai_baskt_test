import streamlit as st
from analysis import analisar_movimento
from skeleton_visualizer import render_side_by_side_with_skeletons, save_skeleton_frame
from feedback import analyze_errors_by_body_part, generate_feedback
from movement_analyzer import MovementAnalyzer
from model_generator import ModelGenerator
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
from terms import terms_text
from reference_processor import ReferenceProcessor, process_all_movements

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

def detectar_frame_saida_bola(keypoints_usuario, threshold=0.08, segundos_fallback=2, fps=30):
    """
    Detecta o frame em que a bola sai da mão do jogador.
    Heurística: considera saída quando o deslocamento do punho direito (keypoint 16) aumenta abruptamente.
    Se não detectar, retorna o frame correspondente a 'segundos_fallback'.
    """
    if keypoints_usuario is None or len(keypoints_usuario) < 2:
        return int(fps * segundos_fallback)
    pos_mao = keypoints_usuario[:, 16, :2]  # (x, y) do punho direito
    diffs = np.linalg.norm(np.diff(pos_mao, axis=0), axis=1)
    for i, d in enumerate(diffs):
        if d > threshold:
            return i + 1  # frame após a mudança
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
    """
    Nova função de análise usando o MovementAnalyzer que compara com modelos de referência.
    """
    # Verificar se os arquivos existem
    if not os.path.exists(user_path):
        st.error("❌ Vídeo do usuário não encontrado!")
        return None
    
    # Mapear tipo de movimento para o formato usado nos modelos
    movement_mapping = {
        "Arremesso": "arremesso",
        "Bandeja": "bandeja",
        "Spin": "spin",
        "Half Spin": "half spin",
        "Crossover": "crossover",
        "Crossover por trás": "crossover por trás",
        "Crossover entre as pernas": "crossover entre as pernas"
    }
    
    movement_type = movement_mapping.get(tipo_movimento)
    if not movement_type:
        st.error(f"❌ Tipo de movimento '{tipo_movimento}' não suportado!")
        return None
    
    # Inicializar o analisador
    analyzer = MovementAnalyzer()
    
    try:
        # Realizar análise completa
        resultado = analyzer.analyze_user_movement(user_path, movement_type)
        
        if not resultado['success']:
            st.error(f"❌ Erro na análise: {resultado['error']}")
            return None
        
        # Exibir resultados
        st.subheader(f"🎯 Score Final: {resultado['final_score']:.1f}/100")
        
        # Visualização do esqueleto sobreposto
        st.subheader("🦴 Visualização do Movimento")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Vídeo Original**")
            st.video(user_path)
        
        with col2:
            # Aqui seria ideal ter uma visualização do esqueleto sobreposto
            # Por enquanto, mostramos informações do movimento
            st.markdown("**Análise do Movimento**")
            st.write(f"**Tipo:** {tipo_movimento}")
            st.write(f"**Frames analisados:** {resultado['analysis_summary']['total_frames']}")
            st.write(f"**Partes do corpo:** {resultado['analysis_summary']['total_parts_analyzed']}")
        
        # Criar timestamp para nomear a pasta
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copiar vídeo para a pasta de resultados
        results_dir = os.path.join("app", "results", nome_usuario)
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy2(user_path, os.path.join(results_dir, "user.mp4"))
        
        # Preparar dados para salvar
        data = {
            "nome_usuario": nome_usuario,
            "movimento": tipo_movimento,
            "score_geral": resultado['final_score'],
            "data": timestamp,
            "video_path": results_dir,
            "feedback": "Análise concluída com sucesso",
            "analysis_result": resultado,
            "frames_criticos": [frame['frame'] for frame in resultado['critical_frames']],
            "partes_com_maior_erro": [part['part'] for part in resultado['top_error_parts']]
        }
        
        # Salvar análise
        save_analysis(data, nome_usuario)
        
        # Exibir detalhes da análise
        with st.expander("📊 Detalhes da Análise", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 Partes com Maior Erro")
                for i, part in enumerate(resultado['top_error_parts'][:3]):
                    error_percent = (part['total_error'] / max([p['total_error'] for p in resultado['top_error_parts']])) * 100
                    st.write(f"{i+1}. **{part['part']}**: {part['total_error']:.3f} ({error_percent:.1f}%)")
                    
                    # Dicas visuais baseadas no erro
                    if part['total_error'] > 0.5:
                        st.warning(f"⚠️ {part['part']} precisa de atenção especial")
                    elif part['total_error'] > 0.3:
                        st.info(f"💡 {part['part']} pode ser melhorado")
                    else:
                        st.success(f"✅ {part['part']} está bem")
            
            with col2:
                st.subheader("⚠️ Frames Críticos")
                for i, frame in enumerate(resultado['critical_frames'][:3]):
                    st.write(f"{i+1}. **Frame {frame['frame']}**: {frame['error']:.3f}")
                    st.write(f"   Parte: {frame['part']}")
                    
                    # Dicas para frames críticos
                    if frame['error'] > 0.8:
                        st.error(f"🚨 Frame {frame['frame']} - Erro crítico")
                    elif frame['error'] > 0.5:
                        st.warning(f"⚠️ Frame {frame['frame']} - Atenção necessária")
                    else:
                        st.info(f"💡 Frame {frame['frame']} - Pode melhorar")
        
        # Exibir resumo estatístico com dicas
        with st.expander("📈 Resumo Estatístico e Dicas", expanded=False):
            summary = resultado['analysis_summary']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total de frames analisados:** {summary['total_frames']}")
                st.write(f"**Partes do corpo analisadas:** {summary['total_parts_analyzed']}")
                st.write(f"**Erro médio:** {summary['average_error']:.3f}")
                st.write(f"**Erro máximo:** {summary['max_error']:.3f}")
                st.write(f"**Erro mínimo:** {summary['min_error']:.3f}")
            
            with col2:
                # Dicas gerais baseadas no score
                if resultado['final_score'] >= 90:
                    st.success("🎉 Excelente! Seu movimento está muito próximo da referência.")
                elif resultado['final_score'] >= 75:
                    st.info("👍 Bom trabalho! Algumas pequenas melhorias podem ser feitas.")
                elif resultado['final_score'] >= 60:
                    st.warning("⚠️ Movimento aceitável, mas há pontos importantes para melhorar.")
                else:
                    st.error("🚨 Atenção! O movimento precisa de melhorias significativas.")
                
                # Dicas específicas
                if summary['average_error'] > 0.5:
                    st.write("💡 **Dica:** Pratique o movimento mais lentamente para melhorar a precisão.")
                if summary['max_error'] > 1.0:
                    st.write("💡 **Dica:** Há momentos específicos que precisam de mais atenção.")
                if len([p for p in resultado['top_error_parts'] if p['total_error'] > 0.5]) > 2:
                    st.write("💡 **Dica:** Foque em melhorar a coordenação geral do movimento.")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")
        return None
    finally:
        analyzer.close()

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
    st.markdown("<p style='text-align: center; color: red;'>Ainda estamos em fase teste, em caso de bug, nos avise !</p>", unsafe_allow_html=True)

    # Expander com os Termos de Uso
    with st.expander("📄 Termos de Uso, Privacidade e Consentimento", expanded=False):
        st.markdown(terms_text)

    # Campo para nome do usuário (antes de tudo)
    usuario_nome = st.text_input("Digite seu nome ou apelido para salvar seu histórico:")
    if not usuario_nome:
        st.warning("Por favor, digite seu nome para continuar.")
        st.stop()
    st.session_state["usuario"] = usuario_nome

    # Novo selectbox para tipo de movimento
    movement_type = st.selectbox(
        "Escolha o tipo de movimento para análise:",
        [
            "Arremesso",
            "Bandeja",
            "Spin",
            "Half Spin",
            "Crossover",
            "Crossover por trás",
            "Crossover entre as pernas"
        ]
    )

    # Checkbox de consentimento
    consentimento = st.checkbox("✅ Li e concordo com os termos de uso e política de privacidade.")
    if not consentimento:
        st.info("Você deve concordar com os termos para enviar vídeos e realizar análises.")

    # Seções principais
    tab1, tab2 = st.tabs(["Análise do Movimento", "Histórico de Análises"])

    with tab1:
        st.header("1️⃣ Análise de Movimento")
        st.markdown("Compare seu movimento com modelos de referência pré-processados.")

        # Upload do vídeo do usuário
        uploaded_file = st.file_uploader("📤 Envie seu vídeo", type=['mp4', 'avi', 'mov'], key="user_video")
        
        if uploaded_file is not None:
            # Salvar vídeo do usuário
            user_video_path = os.path.join("app", "temp", "user.mp4")
            with open(user_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("✅ Vídeo do usuário carregado com sucesso!")
            
            # Mostrar vídeo do usuário
            st.video(user_video_path)
            
            # Mapear tipo de movimento para o formato usado nos modelos
            movement_mapping = {
                "Arremesso": "arremesso",
                "Bandeja": "bandeja",
                "Spin": "spin",
                "Half Spin": "half spin",
                "Crossover": "crossover",
                "Crossover por trás": "crossover por trás",
                "Crossover entre as pernas": "crossover entre as pernas"
            }
            
            movement_type = movement_mapping.get(movement_type)
            
            # Verificar se existe modelo e gerar automaticamente se necessário
            model_path = f"referencias/{movement_type}/modelo_{movement_type}.json"
            model_exists = os.path.exists(model_path)
            
            # Se modelo não existe, tentar gerar automaticamente
            if not model_exists:
                with st.spinner("🔄 Preparando modelo de referência..."):
                    generator = ModelGenerator()
                    success = generator.ensure_model_exists(movement_type)
                    if success:
                        model_exists = True
                        st.success("✅ Modelo gerado automaticamente!")
                    else:
                        st.warning("⚠️ Este tipo de movimento ainda está sendo adicionado. Experimente com outro tipo.")
                        st.stop()
            
            # Botão para iniciar análise (só aparece se modelo existe)
            if model_exists:
                if st.button("🚀 Iniciar Análise", type="primary"):
                    if consentimento:
                        with st.spinner("🔍 Analisando movimento..."):
                            # Realizar análise
                            resultados = analyze_and_visualize(
                                user_video_path, 
                                None,  # Não usado mais
                                usuario_nome, 
                                movement_type
                            )
                            
                            if resultados:
                                st.success("✅ Análise concluída com sucesso!")
                            else:
                                st.error("❌ Erro durante a análise")
                    else:
                        st.error("❌ Você deve concordar com os termos para realizar a análise")

    with tab2:
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

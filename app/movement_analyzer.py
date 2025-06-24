"""
Sistema de Análise de Movimentos de Basquete
============================================

Este módulo implementa a análise completa de movimentos comparando vídeos do usuário
com modelos de referência pré-processados.

FUNCIONALIDADES:
1. Extração de keypoints do vídeo do usuário
2. Cálculo de vetores de movimento frame a frame
3. Comparação com modelo de referência
4. Cálculo de erro médio por parte do corpo
5. Ponderação com desvio padrão da base
6. Geração de score final
7. Identificação de partes com maior erro
8. Detecção de frames críticos

USO:
analyzer = MovementAnalyzer()
resultado = analyzer.analyze_user_movement(user_video_path, movement_type)
"""

import os
import json
import numpy as np
import mediapipe as mp
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from reference_processor import ReferenceProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovementAnalyzer:
    """
    Classe principal para análise de movimentos do usuário comparando com modelos de referência.
    """
    
    def __init__(self):
        """Inicializa o analisador com MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mapeamento dos keypoints do MediaPipe para partes do corpo
        self.body_parts = {
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
        
        self.part_names = list(self.body_parts.keys())
    
    def extract_user_keypoints(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extrai keypoints do vídeo do usuário.
        
        Args:
            video_path: Caminho para o vídeo do usuário
            
        Returns:
            Array numpy com keypoints [frames, partes_corpo, coordenadas_xyz]
        """
        logger.info(f"Extraindo keypoints do vídeo do usuário: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Vídeo não encontrado: {video_path}")
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Erro ao abrir vídeo: {video_path}")
            return None
        
        all_keypoints = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                frame_keypoints = []
                for part_name, landmark_idx in self.body_parts.items():
                    landmark = results.pose_landmarks.landmark[landmark_idx]
                    frame_keypoints.append([landmark.x, landmark.y, landmark.z])
                all_keypoints.append(frame_keypoints)
            else:
                # Se não detectou pose, usar zeros
                frame_keypoints = [[0.0, 0.0, 0.0] for _ in self.body_parts]
                all_keypoints.append(frame_keypoints)
            
            frame_count += 1
        
        cap.release()
        
        if not all_keypoints:
            logger.error("Nenhum keypoint extraído do vídeo do usuário")
            return None
        
        keypoints_array = np.array(all_keypoints)
        logger.info(f"Extraídos {len(keypoints_array)} frames do vídeo do usuário")
        
        return keypoints_array
    
    def calculate_user_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calcula vetores de movimento do usuário frame a frame.
        
        Args:
            keypoints: Array [frames, partes_corpo, xyz] com keypoints do usuário
            
        Returns:
            Array [frames-1, partes_corpo, xyz] com vetores de movimento
        """
        logger.info("Calculando vetores de movimento do usuário")
        
        if len(keypoints) < 2:
            logger.warning("Vídeo muito curto para calcular vetores")
            return np.array([])
        
        vectors = keypoints[1:] - keypoints[:-1]
        logger.info(f"Calculados {len(vectors)} vetores de movimento do usuário")
        
        return vectors
    
    def load_movement_model(self, movement_type: str) -> Optional[Dict]:
        """
        Carrega o modelo de referência para o tipo de movimento especificado.
        
        Args:
            movement_type: Tipo de movimento (arremesso, bandeja, etc.)
            
        Returns:
            Dicionário com o modelo de referência ou None se não encontrado
        """
        model_path = f"referencias/{movement_type}/modelo_{movement_type}.json"
        
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            return None
        
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
            logger.info(f"Modelo carregado: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return None
    
    def calculate_errors_by_part(self, user_keypoints: np.ndarray, user_vectors: np.ndarray, 
                                model: Dict) -> Dict:
        """
        Calcula o erro médio por parte do corpo comparando com o modelo.
        
        Args:
            user_keypoints: Keypoints do usuário [frames, partes_corpo, xyz]
            user_vectors: Vetores do usuário [frames-1, partes_corpo, xyz]
            model: Modelo de referência carregado
            
        Returns:
            Dicionário com erros por parte do corpo
        """
        logger.info("Calculando erros por parte do corpo")
        
        errors_by_part = {}
        
        # Acessar estatísticas do modelo
        model_stats = model.get('statistics', {})
        
        for i, part_name in enumerate(self.part_names):
            if part_name not in model_stats:
                logger.warning(f"Parte {part_name} não encontrada no modelo")
                continue
            
            # Extrair dados da parte do corpo
            part_keypoints = user_keypoints[:, i, :]  # [frames, xyz]
            
            # Obter estatísticas do modelo
            part_stats = model_stats[part_name]
            model_keypoints_mean = np.array(part_stats['keypoints']['mean'])
            model_keypoints_std = np.array(part_stats['keypoints']['std'])
            
            # Calcular erro dos keypoints
            keypoint_errors = np.abs(part_keypoints - model_keypoints_mean)
            
            # Normalizar pelo desvio padrão do modelo (ponderação)
            normalized_errors = keypoint_errors / (model_keypoints_std + 1e-8)  # Evitar divisão por zero
            
            # Calcular erro médio ponderado
            mean_error = np.mean(normalized_errors)
            
            # Calcular erro dos vetores se disponível
            vector_error = 0.0
            if len(user_vectors) > 0 and 'vectors' in part_stats:
                part_vectors = user_vectors[:, i, :]  # [frames-1, xyz]
                model_vectors_mean = np.array(part_stats['vectors']['mean'])
                model_vectors_std = np.array(part_stats['vectors']['std'])
                
                vector_errors = np.abs(part_vectors - model_vectors_mean)
                normalized_vector_errors = vector_errors / (model_vectors_std + 1e-8)
                vector_error = np.mean(normalized_vector_errors)
            
            # Erro total (média ponderada entre keypoints e vetores)
            total_error = (mean_error + vector_error) / 2
            
            errors_by_part[part_name] = {
                'keypoint_error': float(mean_error),
                'vector_error': float(vector_error),
                'total_error': float(total_error),
                'frame_errors': normalized_errors.tolist()  # Para identificar frames críticos
            }
        
        logger.info(f"Erros calculados para {len(errors_by_part)} partes do corpo")
        return errors_by_part
    
    def calculate_final_score(self, errors_by_part: Dict) -> float:
        """
        Calcula o score final baseado no erro médio total ponderado.
        
        Args:
            errors_by_part: Dicionário com erros por parte do corpo
            
        Returns:
            Score final entre 0 e 100
        """
        if not errors_by_part:
            return 0.0
        
        # Calcular erro médio total
        total_errors = [data['total_error'] for data in errors_by_part.values()]
        mean_total_error = np.mean(total_errors)
        
        # Converter para score (0-100, onde 100 é perfeito)
        # Usar função exponencial para penalizar erros maiores
        score = 100 * np.exp(-mean_total_error)
        score = max(0.0, min(100.0, score))  # Limitar entre 0 e 100
        
        logger.info(f"Score final calculado: {score:.2f}")
        return score
    
    def identify_critical_frames(self, errors_by_part: Dict, top_n: int = 5) -> List[Dict]:
        """
        Identifica os frames críticos com maior desvio.
        
        Args:
            errors_by_part: Dicionário com erros por parte do corpo
            top_n: Número de frames críticos a retornar
            
        Returns:
            Lista de dicionários com informações dos frames críticos
        """
        logger.info(f"Identificando {top_n} frames críticos")
        
        # Coletar todos os erros de frame
        all_frame_errors = []
        
        for part_name, data in errors_by_part.items():
            frame_errors = data['frame_errors']
            for frame_idx, error in enumerate(frame_errors):
                # Calcular erro total do frame (média dos eixos x, y, z)
                total_frame_error = np.mean(error)
                all_frame_errors.append({
                    'frame': frame_idx,
                    'part': part_name,
                    'error': total_frame_error,
                    'errors_xyz': error
                })
        
        # Ordenar por erro (maior primeiro)
        all_frame_errors.sort(key=lambda x: x['error'], reverse=True)
        
        # Retornar os top_n frames críticos
        critical_frames = all_frame_errors[:top_n]
        
        logger.info(f"Identificados {len(critical_frames)} frames críticos")
        return critical_frames
    
    def get_body_parts_with_highest_errors(self, errors_by_part: Dict, top_n: int = 5) -> List[Dict]:
        """
        Identifica as partes do corpo com maior erro.
        
        Args:
            errors_by_part: Dicionário com erros por parte do corpo
            top_n: Número de partes a retornar
            
        Returns:
            Lista de dicionários com informações das partes com maior erro
        """
        logger.info(f"Identificando {top_n} partes do corpo com maior erro")
        
        # Converter para lista e ordenar por erro total
        parts_with_errors = []
        for part_name, data in errors_by_part.items():
            parts_with_errors.append({
                'part': part_name,
                'total_error': data['total_error'],
                'keypoint_error': data['keypoint_error'],
                'vector_error': data['vector_error']
            })
        
        # Ordenar por erro total (maior primeiro)
        parts_with_errors.sort(key=lambda x: x['total_error'], reverse=True)
        
        # Retornar as top_n partes
        top_parts = parts_with_errors[:top_n]
        
        logger.info(f"Identificadas {len(top_parts)} partes com maior erro")
        return top_parts
    
    def analyze_user_movement(self, user_video_path: str, movement_type: str) -> Dict:
        """
        Função principal que executa a análise completa do movimento do usuário.
        
        Args:
            user_video_path: Caminho para o vídeo do usuário
            movement_type: Tipo de movimento (arremesso, bandeja, etc.)
            
        Returns:
            Dicionário com resultados completos da análise
        """
        logger.info(f"Iniciando análise do movimento {movement_type}")
        
        # 1. Extrair keypoints do vídeo do usuário
        user_keypoints = self.extract_user_keypoints(user_video_path)
        if user_keypoints is None:
            return {
                'success': False,
                'error': 'Falha ao extrair keypoints do vídeo do usuário'
            }
        
        # 2. Calcular vetores do usuário
        user_vectors = self.calculate_user_vectors(user_keypoints)
        
        # 3. Carregar modelo de referência
        model = self.load_movement_model(movement_type)
        if model is None:
            return {
                'success': False,
                'error': f'Modelo de referência não encontrado para {movement_type}'
            }
        
        # 4. Calcular erros por parte do corpo
        errors_by_part = self.calculate_errors_by_part(user_keypoints, user_vectors, model)
        
        # 5. Calcular score final
        final_score = self.calculate_final_score(errors_by_part)
        
        # 6. Identificar partes com maior erro
        top_error_parts = self.get_body_parts_with_highest_errors(errors_by_part, top_n=5)
        
        # 7. Identificar frames críticos
        critical_frames = self.identify_critical_frames(errors_by_part, top_n=5)
        
        # 8. Gerar resultado final
        result = {
            'success': True,
            'movement_type': movement_type,
            'final_score': final_score,
            'errors_by_part': errors_by_part,
            'top_error_parts': top_error_parts,
            'critical_frames': critical_frames,
            'analysis_summary': {
                'total_frames': len(user_keypoints),
                'total_parts_analyzed': len(errors_by_part),
                'average_error': np.mean([data['total_error'] for data in errors_by_part.values()]),
                'max_error': max([data['total_error'] for data in errors_by_part.values()]),
                'min_error': min([data['total_error'] for data in errors_by_part.values()])
            }
        }
        
        logger.info(f"Análise concluída. Score: {final_score:.2f}")
        return result
    
    def close(self):
        """Libera recursos do MediaPipe."""
        self.pose.close() 
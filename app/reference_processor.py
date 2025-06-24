"""
Sistema de Processamento de Referências para Análise de Movimento de Basquete
================================================================================

Este script processa vídeos de referência para criar modelos estatísticos de movimento.
Cada função tem uma responsabilidade específica e é executada em sequência.

ESTRUTURA:
- extract_keypoints_from_video(): Extrai keypoints de um vídeo
- calculate_motion_vectors(): Calcula vetores de movimento entre frames
- compute_statistics(): Calcula estatísticas (média, desvio, limites)
- process_single_reference(): Processa um vídeo de referência completo
- process_movement_references(): Processa todos os vídeos de um movimento
- save_movement_model(): Salva o modelo estatístico do movimento

USO:
1. Execute process_movement_references("bandeja") para processar todos os vídeos de bandeja
2. Os arquivos serão salvos em /referencias/bandeja/
"""

import os
import json
import numpy as np
import mediapipe as mp
import cv2
from typing import Dict, List, Tuple, Optional
import logging

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceProcessor:
    """
    Classe principal para processamento de vídeos de referência.
    Cada método tem uma responsabilidade específica e bem definida.
    """
    
    def __init__(self):
        """Inicializa o processador com MediaPipe Pose."""
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
    
    def extract_keypoints_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        PASSO 1: Extrair keypoints de um vídeo usando MediaPipe.
        
        Args:
            video_path: Caminho para o vídeo de referência
            
        Returns:
            Array numpy com keypoints [frames, partes_corpo, coordenadas_xyz]
            None se houver erro
        """
        logger.info(f"Extraindo keypoints de: {video_path}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(video_path):
            logger.error(f"Vídeo não encontrado: {video_path}")
            return None
        
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Erro ao abrir vídeo: {video_path}")
            return None
        
        # Lista para armazenar keypoints de todos os frames
        all_keypoints = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Converter para RGB (MediaPipe requer RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar pose
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extrair keypoints das partes do corpo definidas
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
        
        # Liberar recursos
        cap.release()
        
        if not all_keypoints:
            logger.error("Nenhum keypoint extraído do vídeo")
            return None
        
        # Converter para array numpy
        keypoints_array = np.array(all_keypoints)
        logger.info(f"Extraídos {len(keypoints_array)} frames com {len(self.body_parts)} partes do corpo")
        
        return keypoints_array
    
    def calculate_motion_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """
        PASSO 2: Calcular vetores de movimento entre frames consecutivos.
        
        Args:
            keypoints: Array [frames, partes_corpo, xyz] com keypoints
            
        Returns:
            Array [frames-1, partes_corpo, xyz] com vetores de movimento
        """
        logger.info("Calculando vetores de movimento")
        
        if len(keypoints) < 2:
            logger.warning("Vídeo muito curto para calcular vetores")
            return np.array([])
        
        # Calcular diferença entre frames consecutivos
        vectors = keypoints[1:] - keypoints[:-1]
        
        logger.info(f"Calculados {len(vectors)} vetores de movimento")
        return vectors
    
    def compute_statistics(self, keypoints: np.ndarray, vectors: np.ndarray) -> Dict:
        """
        PASSO 3: Calcular estatísticas (média, desvio padrão, limites) para cada parte do corpo.
        
        Args:
            keypoints: Array com keypoints [frames, partes_corpo, xyz]
            vectors: Array com vetores [frames-1, partes_corpo, xyz]
            
        Returns:
            Dicionário com estatísticas por parte do corpo
        """
        logger.info("Calculando estatísticas")
        
        stats = {}
        part_names = list(self.body_parts.keys())
        
        for i, part_name in enumerate(part_names):
            # Estatísticas dos keypoints
            part_keypoints = keypoints[:, i, :]  # [frames, xyz]
            
            # Estatísticas dos vetores (se existirem)
            if len(vectors) > 0:
                part_vectors = vectors[:, i, :]  # [frames-1, xyz]
            else:
                part_vectors = np.array([])
            
            stats[part_name] = {
                'keypoints': {
                    'mean': part_keypoints.mean(axis=0).tolist(),
                    'std': part_keypoints.std(axis=0).tolist(),
                    'min': part_keypoints.min(axis=0).tolist(),
                    'max': part_keypoints.max(axis=0).tolist()
                }
            }
            
            # Adicionar estatísticas dos vetores se existirem
            if len(part_vectors) > 0:
                stats[part_name]['vectors'] = {
                    'mean': part_vectors.mean(axis=0).tolist(),
                    'std': part_vectors.std(axis=0).tolist(),
                    'min': part_vectors.min(axis=0).tolist(),
                    'max': part_vectors.max(axis=0).tolist()
                }
        
        logger.info(f"Estatísticas calculadas para {len(part_names)} partes do corpo")
        return stats
    
    def process_single_reference(self, video_path: str, output_dir: str, ref_name: str) -> bool:
        """
        PASSO 4: Processar um vídeo de referência completo.
        
        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório de saída
            ref_name: Nome da referência (ex: "ref1")
            
        Returns:
            True se processado com sucesso, False caso contrário
        """
        logger.info(f"Processando referência: {ref_name}")
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # PASSO 1: Extrair keypoints
        keypoints = self.extract_keypoints_from_video(video_path)
        if keypoints is None:
            return False
        
        # PASSO 2: Calcular vetores
        vectors = self.calculate_motion_vectors(keypoints)
        
        # Salvar keypoints
        keypoints_file = os.path.join(output_dir, f"{ref_name}_keypoints.json")
        with open(keypoints_file, 'w') as f:
            json.dump(keypoints.tolist(), f, indent=2)
        
        # Salvar vetores
        vectors_file = os.path.join(output_dir, f"{ref_name}_vetores.json")
        with open(vectors_file, 'w') as f:
            json.dump(vectors.tolist(), f, indent=2)
        
        logger.info(f"Salvos: {keypoints_file}, {vectors_file}")
        return True
    
    def process_movement_references(self, movement_name: str) -> Optional[Dict]:
        """
        PASSO 5: Processar todos os vídeos de referência de um movimento.
        
        Args:
            movement_name: Nome do movimento (ex: "bandeja", "arremesso")
            
        Returns:
            Dicionário com estatísticas agregadas do movimento
        """
        logger.info(f"Processando movimento: {movement_name}")
        
        # Caminhos dos diretórios
        videos_dir = os.path.join("movimentos")
        output_dir = os.path.join("referencias", movement_name)
        
        # Mapeamento de nomes de movimento para pastas
        movement_folders = {
            "arremesso": "arremesso",
            "bandeja": "bandeja",
            "spin": "dribles/spin/spin",
            "half spin": "dribles/spin/push cross spin",
            "crossover": "dribles/crossover/simples",
            "crossover por trás": "dribles/crossover/atras_costas",
            "crossover entre as pernas": "dribles/crossover/entre_pernas"
        }
        
        # Encontrar pasta do movimento
        movement_folder = movement_folders.get(movement_name.lower())
        if not movement_folder:
            logger.error(f"Movimento não encontrado: {movement_name}")
            return None
        
        videos_path = os.path.join(videos_dir, movement_folder)
        if not os.path.exists(videos_path):
            logger.error(f"Pasta de vídeos não encontrada: {videos_path}")
            return None
        
        # Listar vídeos de referência
        video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
        if not video_files:
            logger.error(f"Nenhum vídeo encontrado em: {videos_path}")
            return None
        
        logger.info(f"Encontrados {len(video_files)} vídeos para processar")
        
        # Processar cada vídeo
        all_keypoints = []
        all_vectors = []
        
        for i, video_file in enumerate(sorted(video_files)):
            video_path = os.path.join(videos_path, video_file)
            ref_name = f"ref{i+1}"
            
            # Processar vídeo individual
            success = self.process_single_reference(video_path, output_dir, ref_name)
            if not success:
                logger.warning(f"Falha ao processar: {video_file}")
                continue
            
            # Carregar dados para estatísticas agregadas
            keypoints_file = os.path.join(output_dir, f"{ref_name}_keypoints.json")
            vectors_file = os.path.join(output_dir, f"{ref_name}_vetores.json")
            
            with open(keypoints_file, 'r') as f:
                keypoints = np.array(json.load(f))
                all_keypoints.append(keypoints)
            
            with open(vectors_file, 'r') as f:
                vectors = np.array(json.load(f))
                all_vectors.append(vectors)
        
        if not all_keypoints:
            logger.error("Nenhum vídeo foi processado com sucesso")
            return None
        
        # PASSO 6: Calcular estatísticas agregadas
        logger.info("Calculando estatísticas agregadas")
        
        # Concatenar todos os keypoints e vetores
        combined_keypoints = np.concatenate(all_keypoints, axis=0)
        combined_vectors = np.concatenate(all_vectors, axis=0)
        
        # Calcular estatísticas
        movement_stats = self.compute_statistics(combined_keypoints, combined_vectors)
        
        # Adicionar metadados
        movement_model = {
            'movement_name': movement_name,
            'num_references': len(all_keypoints),
            'total_frames': len(combined_keypoints),
            'body_parts': list(self.body_parts.keys()),
            'statistics': movement_stats,
            'created_at': str(np.datetime64('now'))
        }
        
        # Salvar modelo do movimento
        model_file = os.path.join(output_dir, f"modelo_{movement_name}.json")
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(movement_model, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Modelo salvo: {model_file}")
        return movement_model
    
    def load_movement_model(self, movement_name: str) -> Optional[Dict]:
        """
        Carregar modelo de movimento salvo.
        
        Args:
            movement_name: Nome do movimento
            
        Returns:
            Dicionário com o modelo ou None se não encontrado
        """
        model_file = os.path.join("referencias", movement_name, f"modelo_{movement_name}.json")
        
        if not os.path.exists(model_file):
            return None
        
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return None
    
    def close(self):
        """Liberar recursos do MediaPipe."""
        self.pose.close()


def process_all_movements() -> Dict[str, Dict]:
    """
    Processar todos os tipos de movimento disponíveis.
    
    Returns:
        Dicionário com modelos de todos os movimentos processados
    """
    movements = [
        "arremesso",
        "bandeja", 
        "spin",
        "half spin",
        "crossover",
        "crossover por trás",
        "crossover entre as pernas"
    ]
    
    processor = ReferenceProcessor()
    results = {}
    
    try:
        for movement in movements:
            logger.info(f"=== Processando {movement.upper()} ===")
            model = processor.process_movement_references(movement)
            if model:
                results[movement] = model
                logger.info(f"✅ {movement} processado com sucesso")
            else:
                logger.error(f"❌ Falha ao processar {movement}")
    finally:
        processor.close()
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    print("=== Sistema de Processamento de Referências ===")
    
    # Processar um movimento específico
    processor = ReferenceProcessor()
    try:
        model = processor.process_movement_references("bandeja")
        if model:
            print("✅ Bandeja processada com sucesso!")
            print(f"Referências processadas: {model['num_references']}")
        else:
            print("❌ Falha ao processar bandeja")
    finally:
        processor.close() 
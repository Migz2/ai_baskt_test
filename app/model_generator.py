"""
Gerador Automático de Modelos de Referência
===========================================

Este módulo gera automaticamente modelos de referência baseado nos arquivos
de vetores e keypoints existentes, garantindo que o sistema sempre tenha
os modelos necessários.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelGenerator:
    """
    Classe para gerar automaticamente modelos de referência.
    """
    
    def __init__(self):
        """Inicializa o gerador de modelos."""
        self.body_parts = [
            'ombro_esquerdo',
            'ombro_direito', 
            'cotovelo_esquerdo',
            'cotovelo_direito',
            'punho_esquerdo',
            'punho_direito',
            'quadril_esquerdo',
            'quadril_direito',
            'joelho_esquerdo',
            'joelho_direito',
            'tornozelo_esquerdo',
            'tornozelo_direito'
        ]
    
    def load_vectors_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Carrega um arquivo de vetores.
        
        Args:
            file_path: Caminho para o arquivo de vetores
            
        Returns:
            Array numpy com os vetores ou None se erro
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converter para array numpy
            vectors = np.array(data)
            return vectors
        except Exception as e:
            logger.error(f"Erro ao carregar {file_path}: {e}")
            return None
    
    def load_keypoints_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Carrega um arquivo de keypoints.
        
        Args:
            file_path: Caminho para o arquivo de keypoints
            
        Returns:
            Array numpy com os keypoints ou None se erro
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converter para array numpy
            keypoints = np.array(data)
            return keypoints
        except Exception as e:
            logger.error(f"Erro ao carregar {file_path}: {e}")
            return None
    
    def calculate_statistics(self, data_list: List[np.ndarray]) -> Dict:
        """
        Calcula estatísticas (média e desvio padrão) de uma lista de arrays.
        
        Args:
            data_list: Lista de arrays numpy
            
        Returns:
            Dicionário com média e desvio padrão
        """
        if not data_list:
            return {'mean': [0.0, 0.0, 0.0], 'std': [0.0, 0.0, 0.0]}
        
        # Concatenar todos os arrays
        all_data = np.concatenate(data_list, axis=0)
        
        # Calcular estatísticas
        mean = np.mean(all_data, axis=0).tolist()
        std = np.std(all_data, axis=0).tolist()
        
        return {
            'mean': mean,
            'std': std
        }
    
    def generate_movement_model(self, movement_name: str) -> Optional[Dict]:
        """
        Gera modelo de referência para um movimento específico.
        
        Args:
            movement_name: Nome do movimento (ex: 'arremesso', 'bandeja')
            
        Returns:
            Dicionário com o modelo gerado ou None se erro
        """
        logger.info(f"Gerando modelo para movimento: {movement_name}")
        
        movement_dir = f"referencias/{movement_name}"
        if not os.path.exists(movement_dir):
            logger.error(f"Diretório não encontrado: {movement_dir}")
            return None
        
        # Buscar arquivos de vetores
        vectors_files = []
        keypoints_files = []
        
        for file in os.listdir(movement_dir):
            if file.endswith('_vetores.json'):
                vectors_files.append(os.path.join(movement_dir, file))
            elif file.endswith('_keypoints.json'):
                keypoints_files.append(os.path.join(movement_dir, file))
        
        if not vectors_files:
            logger.error(f"Nenhum arquivo de vetores encontrado para {movement_name}")
            return None
        
        logger.info(f"Encontrados {len(vectors_files)} arquivos de vetores")
        
        # Carregar vetores
        all_vectors = []
        for file_path in vectors_files:
            vectors = self.load_vectors_file(file_path)
            if vectors is not None:
                all_vectors.append(vectors)
        
        # Carregar keypoints
        all_keypoints = []
        for file_path in keypoints_files:
            keypoints = self.load_keypoints_file(file_path)
            if keypoints is not None:
                all_keypoints.append(keypoints)
        
        if not all_vectors:
            logger.error(f"Nenhum vetor válido carregado para {movement_name}")
            return None
        
        # Gerar modelo
        model = {
            'movement_name': movement_name,
            'num_references': len(all_vectors),
            'total_frames': sum(len(v) for v in all_vectors),
            'body_parts': self.body_parts,
            'statistics': {}
        }
        
        # Calcular estatísticas para cada parte do corpo
        for i, part_name in enumerate(self.body_parts):
            # Estatísticas dos keypoints
            part_keypoints = []
            for keypoints in all_keypoints:
                if len(keypoints) > 0 and keypoints.shape[1] > i:
                    part_keypoints.append(keypoints[:, i, :])
            
            # Estatísticas dos vetores
            part_vectors = []
            for vectors in all_vectors:
                if len(vectors) > 0 and vectors.shape[1] > i:
                    part_vectors.append(vectors[:, i, :])
            
            model['statistics'][part_name] = {
                'keypoints': self.calculate_statistics(part_keypoints)
            }
            
            if part_vectors:
                model['statistics'][part_name]['vectors'] = self.calculate_statistics(part_vectors)
        
        logger.info(f"Modelo gerado com sucesso para {movement_name}")
        return model
    
    def save_model(self, model: Dict, movement_name: str) -> bool:
        """
        Salva o modelo gerado em arquivo JSON.
        
        Args:
            model: Modelo a ser salvo
            movement_name: Nome do movimento
            
        Returns:
            True se salvo com sucesso, False caso contrário
        """
        try:
            movement_dir = f"referencias/{movement_name}"
            os.makedirs(movement_dir, exist_ok=True)
            
            model_path = f"{movement_dir}/modelo_{movement_name}.json"
            
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Modelo salvo em: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False
    
    def ensure_model_exists(self, movement_name: str) -> bool:
        """
        Garante que o modelo existe, gerando automaticamente se necessário.
        
        Args:
            movement_name: Nome do movimento
            
        Returns:
            True se modelo existe ou foi gerado com sucesso
        """
        model_path = f"referencias/{movement_name}/modelo_{movement_name}.json"
        
        if os.path.exists(model_path):
            logger.info(f"Modelo já existe: {model_path}")
            return True
        
        logger.info(f"Modelo não encontrado, gerando automaticamente: {model_path}")
        
        # Gerar modelo
        model = self.generate_movement_model(movement_name)
        if model is None:
            logger.error(f"Falha ao gerar modelo para {movement_name}")
            return False
        
        # Salvar modelo
        success = self.save_model(model, movement_name)
        if success:
            logger.info(f"Modelo gerado e salvo com sucesso para {movement_name}")
        
        return success
    
    def ensure_all_models(self) -> Dict[str, bool]:
        """
        Garante que todos os modelos existem.
        
        Returns:
            Dicionário com status de cada modelo
        """
        movements = [
            'arremesso',
            'bandeja',
            'spin', 
            'half spin',
            'crossover',
            'crossover por trás',
            'crossover entre as pernas'
        ]
        
        results = {}
        
        for movement in movements:
            results[movement] = self.ensure_model_exists(movement)
        
        return results

def generate_missing_models():
    """
    Função principal para gerar modelos ausentes.
    """
    generator = ModelGenerator()
    results = generator.ensure_all_models()
    
    print("🔧 Verificando e gerando modelos ausentes...")
    
    for movement, success in results.items():
        if success:
            print(f"✅ {movement}: modelo disponível")
        else:
            print(f"❌ {movement}: falha ao gerar modelo")
    
    return results

if __name__ == "__main__":
    generate_missing_models() 
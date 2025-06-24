# Sistema de Análise de Movimentos de Basquete

## Visão Geral

O sistema de análise de movimentos implementa uma comparação completa entre vídeos do usuário e modelos de referência pré-processados. O sistema extrai keypoints, calcula vetores de movimento, compara com modelos estatísticos e gera scores detalhados.

## Funcionalidades Principais

### 1. Extração de Keypoints
- Extrai keypoints de 12 partes do corpo usando MediaPipe Pose
- Coordenadas 3D (x, y, z) para cada parte
- Processamento frame a frame

### 2. Cálculo de Vetores de Movimento
- Calcula vetores de movimento entre frames consecutivos
- Representa a velocidade e direção de cada parte do corpo
- Normalização temporal para comparação

### 3. Comparação com Modelos de Referência
- Carrega modelos estatísticos pré-processados
- Compara keypoints e vetores do usuário com o modelo
- Ponderação com desvio padrão da base de referência

### 4. Análise de Erros
- Erro médio por parte do corpo
- Normalização pelo desvio padrão do modelo
- Identificação de partes com maior erro

### 5. Geração de Scores
- Score final baseado no erro médio ponderado
- Função exponencial para penalizar erros maiores
- Escala de 0-100 (100 = perfeito)

### 6. Detecção de Frames Críticos
- Identifica frames com maior desvio
- Ranking dos frames mais problemáticos
- Análise por parte do corpo

## Estrutura do Sistema

### MovementAnalyzer Class

```python
class MovementAnalyzer:
    def __init__(self):
        # Inicializa MediaPipe Pose
        # Define mapeamento de partes do corpo
    
    def extract_user_keypoints(self, video_path):
        # Extrai keypoints do vídeo do usuário
    
    def calculate_user_vectors(self, keypoints):
        # Calcula vetores de movimento
    
    def load_movement_model(self, movement_type):
        # Carrega modelo de referência
    
    def calculate_errors_by_part(self, user_keypoints, user_vectors, model):
        # Calcula erros por parte do corpo
    
    def calculate_final_score(self, errors_by_part):
        # Calcula score final
    
    def identify_critical_frames(self, errors_by_part):
        # Identifica frames críticos
    
    def get_body_parts_with_highest_errors(self, errors_by_part):
        # Identifica partes com maior erro
    
    def analyze_user_movement(self, user_video_path, movement_type):
        # Função principal de análise
```

## Partes do Corpo Analisadas

1. **Ombros**: ombro_esquerdo, ombro_direito
2. **Cotovelos**: cotovelo_esquerdo, cotovelo_direito
3. **Punhos**: punho_esquerdo, punho_direito
4. **Quadris**: quadril_esquerdo, quadril_direito
5. **Joelhos**: joelho_esquerdo, joelho_direito
6. **Tornozelos**: tornozelo_esquerdo, tornozelo_direito

## Tipos de Movimento Suportados

- **Arremesso**: Movimento de arremesso
- **Bandeja**: Movimento de bandeja
- **Spin**: Drible com giro
- **Half Spin**: Meio giro
- **Crossover**: Crossover simples
- **Crossover por trás**: Crossover por trás das costas
- **Crossover entre as pernas**: Crossover entre as pernas

## Como Usar

### 1. Preparação dos Modelos

Primeiro, processe os vídeos de referência para criar os modelos:

```python
from reference_processor import ReferenceProcessor

processor = ReferenceProcessor()
model = processor.process_movement_references("arremesso")
```

### 2. Análise de Movimento

```python
from movement_analyzer import MovementAnalyzer

analyzer = MovementAnalyzer()
resultado = analyzer.analyze_user_movement("user_video.mp4", "arremesso")
```

### 3. Interpretação dos Resultados

```python
if resultado['success']:
    print(f"Score: {resultado['final_score']:.1f}/100")
    
    # Partes com maior erro
    for part in resultado['top_error_parts']:
        print(f"{part['part']}: {part['total_error']:.3f}")
    
    # Frames críticos
    for frame in resultado['critical_frames']:
        print(f"Frame {frame['frame']}: {frame['error']:.3f}")
```

## Estrutura dos Resultados

```json
{
    "success": true,
    "movement_type": "arremesso",
    "final_score": 85.5,
    "errors_by_part": {
        "ombro_esquerdo": {
            "keypoint_error": 0.123,
            "vector_error": 0.456,
            "total_error": 0.289,
            "frame_errors": [...]
        }
    },
    "top_error_parts": [
        {
            "part": "punho_direito",
            "total_error": 0.567,
            "keypoint_error": 0.234,
            "vector_error": 0.789
        }
    ],
    "critical_frames": [
        {
            "frame": 15,
            "part": "punho_direito",
            "error": 0.789,
            "errors_xyz": [0.1, 0.2, 0.3]
        }
    ],
    "analysis_summary": {
        "total_frames": 120,
        "total_parts_analyzed": 12,
        "average_error": 0.345,
        "max_error": 0.789,
        "min_error": 0.123
    }
}
```

## Interface Streamlit

### Aba "Análise do Movimento"

1. **Seleção de Tipo**: Escolha o tipo de movimento
2. **Upload de Vídeo**: Envie seu vídeo
3. **Verificação de Modelo**: Sistema verifica se existe modelo
4. **Análise**: Clique em "Iniciar Análise"
5. **Resultados**: Visualize score, partes com erro e frames críticos

### Aba "Processamento de Referências"

1. **Processar Movimento Específico**: Cria modelo para um movimento
2. **Processar Todos**: Cria modelos para todos os movimentos
3. **Verificação**: Mostra modelos existentes

## Arquivos do Sistema

- `movement_analyzer.py`: Classe principal de análise
- `reference_processor.py`: Processamento de vídeos de referência
- `main.py`: Interface Streamlit
- `test_movement_analysis.py`: Script de teste

## Dependências

- MediaPipe Pose
- OpenCV
- NumPy
- Streamlit
- JSON

## Fluxo de Trabalho

1. **Preparação**: Processe vídeos de referência para criar modelos
2. **Análise**: Usuário envia vídeo e seleciona tipo de movimento
3. **Processamento**: Sistema extrai keypoints e calcula vetores
4. **Comparação**: Compara com modelo de referência
5. **Resultados**: Gera score, identifica erros e frames críticos
6. **Feedback**: Apresenta resultados detalhados ao usuário

## Vantagens do Sistema

- **Precisão**: Análise baseada em modelos estatísticos
- **Detalhamento**: Erro por parte do corpo e frame
- **Escalabilidade**: Fácil adição de novos tipos de movimento
- **Usabilidade**: Interface intuitiva
- **Robustez**: Tratamento de erros e validações

## Limitações

- Requer vídeos com boa qualidade
- Depende da detecção do MediaPipe
- Necessita de modelos pré-processados
- Análise em tempo real limitada

## Próximos Passos

- Implementar análise em tempo real
- Adicionar mais tipos de movimento
- Melhorar algoritmos de comparação
- Adicionar feedback personalizado
- Implementar histórico de progresso 
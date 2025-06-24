# Sistema de Processamento de Referências - Basquete

## 📋 Visão Geral

Este sistema processa vídeos de referência para criar modelos estatísticos de movimento de basquete. Cada movimento é analisado e seus padrões são salvos em arquivos JSON para uso posterior na análise de vídeos do usuário.

## 🎯 Objetivo

Criar modelos de referência que contenham:
- **Keypoints** extraídos de cada frame
- **Vetores de movimento** entre frames consecutivos  
- **Estatísticas** (média, desvio padrão, limites) por parte do corpo
- **Modelo agregado** com padrões do movimento

## 📁 Estrutura de Arquivos

```
/referencias/{movimento}/
├── ref1_keypoints.json      # Keypoints do vídeo 1
├── ref1_vetores.json        # Vetores do vídeo 1
├── ref2_keypoints.json      # Keypoints do vídeo 2
├── ref2_vetores.json        # Vetores do vídeo 2
├── ...
├── modelo_{movimento}.json  # Modelo estatístico do movimento
```

## 🔧 Como Usar

### 1. Processar um Movimento Específico

```python
from reference_processor import ReferenceProcessor

# Criar processador
processor = ReferenceProcessor()

# Processar movimento "bandeja"
model = processor.process_movement_references("bandeja")

# Verificar resultado
if model:
    print(f"✅ {model['num_references']} referências processadas")
    print(f"📊 {model['total_frames']} frames analisados")

# Liberar recursos
processor.close()
```

### 2. Processar Todos os Movimentos

```python
from reference_processor import process_all_movements

# Processar todos os movimentos disponíveis
results = process_all_movements()

# Verificar resultados
for movement, model in results.items():
    print(f"✅ {movement}: {model['num_references']} referências")
```

### 3. Carregar Modelo Existente

```python
from reference_processor import ReferenceProcessor

processor = ReferenceProcessor()
model = processor.load_movement_model("bandeja")

if model:
    print(f"📊 Estatísticas para {len(model['statistics'])} partes do corpo")
```

## 🧪 Testes

Execute os testes para verificar se o sistema está funcionando:

```bash
cd app
python test_reference_processing.py
```

Os testes verificam:
- ✅ Extração de keypoints
- ✅ Cálculo de vetores
- ✅ Cálculo de estatísticas
- ✅ Processamento completo
- ✅ Geração de arquivos
- ✅ Carregamento de modelos

## 📊 Tipos de Movimento Suportados

| Movimento | Pasta de Vídeos |
|-----------|-----------------|
| arremesso | movimentos/arremesso/ |
| bandeja | movimentos/bandeja/ |
| spin | movimentos/dribles/spin/spin/ |
| half spin | movimentos/dribles/spin/push cross spin/ |
| crossover | movimentos/dribles/crossover/simples/ |
| crossover por trás | movimentos/dribles/crossover/atras_costas/ |
| crossover entre as pernas | movimentos/dribles/crossover/entre_pernas/ |

## 🔍 Partes do Corpo Analisadas

O sistema extrai keypoints para 12 partes do corpo:

1. **Ombros**: esquerdo (11), direito (12)
2. **Cotovelos**: esquerdo (13), direito (14)
3. **Punhos**: esquerdo (15), direito (16)
4. **Quadris**: esquerdo (23), direito (24)
5. **Joelhos**: esquerdo (25), direito (26)
6. **Tornozelos**: esquerdo (27), direito (28)

## 📈 Estatísticas Calculadas

Para cada parte do corpo, o sistema calcula:

### Keypoints
- **Média**: posição média em x, y, z
- **Desvio Padrão**: variação da posição
- **Mínimo**: posição mínima em cada eixo
- **Máximo**: posição máxima em cada eixo

### Vetores de Movimento
- **Média**: velocidade média de movimento
- **Desvio Padrão**: variação da velocidade
- **Mínimo**: velocidade mínima
- **Máximo**: velocidade máxima

## 🚀 Integração com Streamlit

O sistema está integrado ao aplicativo Streamlit com:

- **Aba "Processamento de Referências"**
- **Processamento individual** por movimento
- **Processamento em lote** de todos os movimentos
- **Verificação** de modelos existentes
- **Interface visual** para acompanhar o progresso

## ⚠️ Requisitos

- **MediaPipe**: Para extração de pose
- **OpenCV**: Para processamento de vídeo
- **NumPy**: Para cálculos matemáticos
- **Vídeos de referência**: Em formato MP4 na pasta `movimentos/`

## 🔧 Solução de Problemas

### Erro: "Vídeo não encontrado"
- Verifique se os vídeos estão na pasta correta
- Confirme se os nomes dos movimentos estão corretos

### Erro: "Nenhum keypoint extraído"
- Verifique se o vídeo tem qualidade suficiente
- Confirme se há pessoas visíveis no vídeo
- Teste com vídeos mais curtos primeiro

### Erro: "Falha no processamento"
- Verifique se há espaço em disco
- Confirme permissões de escrita na pasta
- Teste com um movimento por vez

## 📝 Logs

O sistema gera logs detalhados para debug:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Logs incluem:
- Progresso do processamento
- Número de frames processados
- Estatísticas calculadas
- Arquivos salvos
- Erros encontrados

## 🎯 Próximos Passos

1. **Processar todos os movimentos** disponíveis
2. **Verificar qualidade** dos modelos gerados
3. **Integrar** com o sistema de análise
4. **Testar** com vídeos do usuário
5. **Ajustar** parâmetros se necessário 
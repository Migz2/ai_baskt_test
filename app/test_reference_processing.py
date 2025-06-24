"""
Script de Teste para o Sistema de Processamento de Referências
==============================================================

Este script testa cada função do ReferenceProcessor de forma isolada,
verificando se o processamento está funcionando corretamente.

EXECUÇÃO:
1. python test_reference_processing.py

RESULTADOS:
- Testa extração de keypoints
- Testa cálculo de vetores
- Testa cálculo de estatísticas
- Testa processamento completo
- Verifica arquivos gerados
"""

import os
import json
import numpy as np
from reference_processor import ReferenceProcessor
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keypoints_extraction():
    """Teste 1: Extração de keypoints de um vídeo."""
    print("\n=== TESTE 1: Extração de Keypoints ===")
    
    processor = ReferenceProcessor()
    
    # Procurar um vídeo de teste
    test_video = None
    for root, dirs, files in os.walk("movimentos"):
        for file in files:
            if file.endswith('.mp4'):
                test_video = os.path.join(root, file)
                break
        if test_video:
            break
    
    if not test_video:
        print("❌ Nenhum vídeo encontrado para teste")
        return False
    
    print(f"Vídeo de teste: {test_video}")
    
    # Extrair keypoints
    keypoints = processor.extract_keypoints_from_video(test_video)
    
    if keypoints is None:
        print("❌ Falha na extração de keypoints")
        return False
    
    print(f"✅ Keypoints extraídos: {keypoints.shape}")
    print(f"   - Frames: {keypoints.shape[0]}")
    print(f"   - Partes do corpo: {keypoints.shape[1]}")
    print(f"   - Coordenadas: {keypoints.shape[2]}")
    
    processor.close()
    return True

def test_vectors_calculation():
    """Teste 2: Cálculo de vetores de movimento."""
    print("\n=== TESTE 2: Cálculo de Vetores ===")
    
    processor = ReferenceProcessor()
    
    # Criar dados de teste
    test_keypoints = np.random.rand(10, 12, 3)  # 10 frames, 12 partes, 3 coordenadas
    
    # Calcular vetores
    vectors = processor.calculate_motion_vectors(test_keypoints)
    
    if len(vectors) == 0:
        print("❌ Falha no cálculo de vetores")
        return False
    
    print(f"✅ Vetores calculados: {vectors.shape}")
    print(f"   - Frames: {vectors.shape[0]}")
    print(f"   - Partes do corpo: {vectors.shape[1]}")
    print(f"   - Coordenadas: {vectors.shape[2]}")
    
    processor.close()
    return True

def test_statistics_computation():
    """Teste 3: Cálculo de estatísticas."""
    print("\n=== TESTE 3: Cálculo de Estatísticas ===")
    
    processor = ReferenceProcessor()
    
    # Criar dados de teste
    test_keypoints = np.random.rand(20, 12, 3)
    test_vectors = np.random.rand(19, 12, 3)
    
    # Calcular estatísticas
    stats = processor.compute_statistics(test_keypoints, test_vectors)
    
    if not stats:
        print("❌ Falha no cálculo de estatísticas")
        return False
    
    print(f"✅ Estatísticas calculadas para {len(stats)} partes do corpo")
    
    # Verificar estrutura
    first_part = list(stats.keys())[0]
    print(f"   - Exemplo ({first_part}):")
    print(f"     Keypoints: {len(stats[first_part]['keypoints'])} estatísticas")
    print(f"     Vetores: {len(stats[first_part]['vectors'])} estatísticas")
    
    processor.close()
    return True

def test_single_reference_processing():
    """Teste 4: Processamento de uma referência completa."""
    print("\n=== TESTE 4: Processamento de Referência ===")
    
    processor = ReferenceProcessor()
    
    # Procurar um vídeo de teste
    test_video = None
    for root, dirs, files in os.walk("movimentos"):
        for file in files:
            if file.endswith('.mp4'):
                test_video = os.path.join(root, file)
                break
        if test_video:
            break
    
    if not test_video:
        print("❌ Nenhum vídeo encontrado para teste")
        return False
    
    # Processar referência
    output_dir = "test_output"
    success = processor.process_single_reference(test_video, output_dir, "test_ref")
    
    if not success:
        print("❌ Falha no processamento da referência")
        return False
    
    # Verificar arquivos gerados
    keypoints_file = os.path.join(output_dir, "test_ref_keypoints.json")
    vectors_file = os.path.join(output_dir, "test_ref_vetores.json")
    
    if not os.path.exists(keypoints_file):
        print("❌ Arquivo de keypoints não foi criado")
        return False
    
    if not os.path.exists(vectors_file):
        print("❌ Arquivo de vetores não foi criado")
        return False
    
    print("✅ Referência processada com sucesso")
    print(f"   - Keypoints: {keypoints_file}")
    print(f"   - Vetores: {vectors_file}")
    
    # Limpar arquivos de teste
    os.remove(keypoints_file)
    os.remove(vectors_file)
    os.rmdir(output_dir)
    
    processor.close()
    return True

def test_movement_processing():
    """Teste 5: Processamento de um movimento completo."""
    print("\n=== TESTE 5: Processamento de Movimento ===")
    
    processor = ReferenceProcessor()
    
    # Processar movimento "bandeja" (se existir)
    model = processor.process_movement_references("bandeja")
    
    if not model:
        print("❌ Falha no processamento do movimento")
        return False
    
    print("✅ Movimento processado com sucesso")
    print(f"   - Nome: {model['movement_name']}")
    print(f"   - Referências: {model['num_references']}")
    print(f"   - Frames totais: {model['total_frames']}")
    print(f"   - Partes do corpo: {len(model['body_parts'])}")
    
    # Verificar arquivo do modelo
    model_file = os.path.join("referencias", "bandeja", "modelo_bandeja.json")
    if os.path.exists(model_file):
        print(f"   - Modelo salvo: {model_file}")
    else:
        print("❌ Arquivo do modelo não foi criado")
        return False
    
    processor.close()
    return True

def test_model_loading():
    """Teste 6: Carregamento de modelo salvo."""
    print("\n=== TESTE 6: Carregamento de Modelo ===")
    
    processor = ReferenceProcessor()
    
    # Carregar modelo
    model = processor.load_movement_model("bandeja")
    
    if not model:
        print("❌ Falha no carregamento do modelo")
        return False
    
    print("✅ Modelo carregado com sucesso")
    print(f"   - Nome: {model['movement_name']}")
    print(f"   - Referências: {model['num_references']}")
    print(f"   - Estatísticas: {len(model['statistics'])} partes do corpo")
    
    processor.close()
    return True

def run_all_tests():
    """Executar todos os testes."""
    print("🧪 INICIANDO TESTES DO SISTEMA DE PROCESSAMENTO")
    print("=" * 60)
    
    tests = [
        ("Extração de Keypoints", test_keypoints_extraction),
        ("Cálculo de Vetores", test_vectors_calculation),
        ("Cálculo de Estatísticas", test_statistics_computation),
        ("Processamento de Referência", test_single_reference_processing),
        ("Processamento de Movimento", test_movement_processing),
        ("Carregamento de Modelo", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro no teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema funcionando corretamente.")
    else:
        print("⚠️  Alguns testes falharam. Verifique os logs acima.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 
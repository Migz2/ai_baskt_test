"""
Script de Teste para o Sistema de Análise de Movimentos
======================================================

Este script testa o sistema completo de análise de movimentos.
"""

import os
import sys
import json
from movement_analyzer import MovementAnalyzer

def test_movement_analyzer():
    """Testa o sistema de análise de movimentos."""
    print("🧪 Testando Sistema de Análise de Movimentos")
    print("=" * 50)
    
    # Verificar se existem modelos de referência
    movement_types = [
        "arremesso", "bandeja", "spin", "half spin",
        "crossover", "crossover por trás", "crossover entre as pernas"
    ]
    
    existing_models = []
    for movement in movement_types:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        if os.path.exists(model_path):
            existing_models.append(movement)
            print(f"✅ Modelo encontrado: {movement}")
        else:
            print(f"❌ Modelo não encontrado: {movement}")
    
    if not existing_models:
        print("\n⚠️ Nenhum modelo de referência encontrado!")
        print("💡 Execute primeiro o processamento de referências.")
        return False
    
    # Testar análise com um modelo existente
    test_movement = existing_models[0]
    print(f"\n🔍 Testando análise com modelo: {test_movement}")
    
    # Verificar se existe um vídeo de teste
    test_video_path = "temp/user_video.mp4"
    if not os.path.exists(test_video_path):
        print(f"❌ Vídeo de teste não encontrado: {test_video_path}")
        print("💡 Coloque um vídeo de teste em temp/user_video.mp4")
        return False
    
    # Inicializar analisador
    analyzer = MovementAnalyzer()
    
    try:
        # Realizar análise
        print("🔄 Realizando análise...")
        resultado = analyzer.analyze_user_movement(test_video_path, test_movement)
        
        if resultado['success']:
            print("✅ Análise concluída com sucesso!")
            print(f"🎯 Score Final: {resultado['final_score']:.1f}/100")
            
            # Mostrar partes com maior erro
            print("\n🔴 Partes com Maior Erro:")
            for i, part in enumerate(resultado['top_error_parts'][:3]):
                print(f"  {i+1}. {part['part']}: {part['total_error']:.3f}")
            
            # Mostrar frames críticos
            print("\n⚠️ Frames Críticos:")
            for i, frame in enumerate(resultado['critical_frames'][:3]):
                print(f"  {i+1}. Frame {frame['frame']}: {frame['error']:.3f}")
            
            # Mostrar resumo
            summary = resultado['analysis_summary']
            print(f"\n📊 Resumo:")
            print(f"  - Frames analisados: {summary['total_frames']}")
            print(f"  - Partes analisadas: {summary['total_parts_analyzed']}")
            print(f"  - Erro médio: {summary['average_error']:.3f}")
            
            return True
        else:
            print(f"❌ Erro na análise: {resultado['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        return False
    finally:
        analyzer.close()

def test_model_loading():
    """Testa o carregamento de modelos."""
    print("\n📂 Testando Carregamento de Modelos")
    print("=" * 40)
    
    analyzer = MovementAnalyzer()
    
    movement_types = [
        "arremesso", "bandeja", "spin", "half spin",
        "crossover", "crossover por trás", "crossover entre as pernas"
    ]
    
    for movement in movement_types:
        model = analyzer.load_movement_model(movement)
        if model:
            print(f"✅ Modelo carregado: {movement}")
            # Verificar estrutura do modelo
            if 'statistics' in model and 'ombro_esquerdo' in model['statistics']:
                print(f"  - Estrutura válida: {len(model['statistics'])} partes do corpo")
            else:
                print(f"  ⚠️ Estrutura inválida")
        else:
            print(f"❌ Falha ao carregar: {movement}")
    
    analyzer.close()

def main():
    """Função principal do teste."""
    print("🚀 Iniciando Testes do Sistema de Análise")
    print("=" * 60)
    
    # Teste 1: Carregamento de modelos
    test_model_loading()
    
    # Teste 2: Análise completa
    success = test_movement_analyzer()
    
    if success:
        print("\n🎉 Todos os testes passaram!")
    else:
        print("\n❌ Alguns testes falharam.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 
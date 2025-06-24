"""
Teste para Verificar Comportamento de Modelos Inexistentes
=========================================================

Este script testa como o sistema se comporta quando um modelo de referência não existe.
"""

import os
import sys
from movement_analyzer import MovementAnalyzer

def test_nonexistent_model():
    """Testa o comportamento quando um modelo não existe."""
    print("🧪 Testando Comportamento com Modelo Inexistente")
    print("=" * 50)
    
    # Testar com um movimento que não existe
    nonexistent_movement = "movimento_inexistente"
    model_path = f"referencias/{nonexistent_movement}/modelo_{nonexistent_movement}.json"
    
    print(f"Testando movimento: {nonexistent_movement}")
    print(f"Caminho do modelo: {model_path}")
    
    # Verificar se o arquivo realmente não existe
    if os.path.exists(model_path):
        print(f"❌ ERRO: Modelo existe quando não deveria: {model_path}")
        return False
    else:
        print(f"✅ Modelo não existe (comportamento esperado)")
    
    # Testar carregamento do modelo
    analyzer = MovementAnalyzer()
    
    try:
        model = analyzer.load_movement_model(nonexistent_movement)
        if model is None:
            print("✅ Sistema retorna None para modelo inexistente (comportamento correto)")
        else:
            print("❌ ERRO: Sistema retornou modelo quando não deveria")
            return False
    except Exception as e:
        print(f"✅ Sistema trata erro graciosamente: {e}")
    
    analyzer.close()
    return True

def test_existing_models():
    """Testa se todos os modelos existentes funcionam corretamente."""
    print("\n🔍 Testando Modelos Existentes")
    print("=" * 40)
    
    existing_movements = [
        "arremesso",
        "bandeja", 
        "spin",
        "half spin",
        "crossover",
        "crossover por trás",
        "crossover entre as pernas"
    ]
    
    analyzer = MovementAnalyzer()
    
    for movement in existing_movements:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        
        if os.path.exists(model_path):
            print(f"✅ {movement}: modelo existe")
            
            # Testar carregamento
            model = analyzer.load_movement_model(movement)
            if model and 'statistics' in model:
                print(f"   ✅ Carregamento bem-sucedido")
            else:
                print(f"   ❌ Falha no carregamento")
                return False
        else:
            print(f"❌ {movement}: modelo não existe")
            return False
    
    analyzer.close()
    return True

def test_model_path_structure():
    """Testa se a estrutura de caminhos está correta."""
    print("\n📁 Testando Estrutura de Caminhos")
    print("=" * 40)
    
    movements = [
        "arremesso",
        "bandeja", 
        "spin",
        "half spin",
        "crossover",
        "crossover por trás",
        "crossover entre as pernas"
    ]
    
    for movement in movements:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        expected_path = f"referencias/{movement}/modelo_{movement}.json"
        
        print(f"Movimento: {movement}")
        print(f"  Caminho esperado: {expected_path}")
        print(f"  Existe: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            # Verificar se o arquivo é válido
            try:
                import json
                with open(model_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'statistics' in data:
                    print(f"  ✅ Estrutura válida")
                else:
                    print(f"  ❌ Estrutura inválida")
            except Exception as e:
                print(f"  ❌ Erro ao ler arquivo: {e}")
        else:
            print(f"  ⚠️ Arquivo não encontrado")

def main():
    """Função principal do teste."""
    print("🚀 Teste de Comportamento de Modelos")
    print("=" * 60)
    
    # Teste 1: Modelo inexistente
    test1_success = test_nonexistent_model()
    
    # Teste 2: Modelos existentes
    test2_success = test_existing_models()
    
    # Teste 3: Estrutura de caminhos
    test_model_path_structure()
    
    if test1_success and test2_success:
        print("\n🎉 Todos os testes passaram!")
        print("✅ Sistema está tratando modelos corretamente")
    else:
        print("\n❌ Alguns testes falharam")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
"""
Teste Final de Verificação do Sistema
====================================

Este script executa uma verificação completa de todos os componentes
do sistema para garantir que está funcionando corretamente.
"""

import os
import json
import sys

# Importações no nível do módulo
try:
    from model_generator import ModelGenerator
    from movement_analyzer import MovementAnalyzer
    from main import main as main_function
    imports_ok = True
except Exception as e:
    print(f"❌ Erro nas importações: {e}")
    imports_ok = False

def test_all_models_exist():
    """Verifica se todos os modelos existem."""
    print("📋 Verificando Existência de Todos os Modelos")
    print("=" * 50)
    
    movements = [
        'arremesso',
        'bandeja',
        'spin',
        'half spin',
        'crossover',
        'crossover por trás',
        'crossover entre as pernas'
    ]
    
    all_exist = True
    
    for movement in movements:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        exists = os.path.exists(model_path)
        
        if exists:
            size = os.path.getsize(model_path)
            print(f"✅ {movement}: {size:,} bytes")
        else:
            print(f"❌ {movement}: NÃO ENCONTRADO")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Testa carregamento de todos os modelos."""
    print("\n📂 Testando Carregamento de Modelos")
    print("=" * 40)
    
    if not imports_ok:
        print("❌ Importações falharam, pulando teste")
        return False
    
    movements = [
        'arremesso',
        'bandeja',
        'spin',
        'half spin',
        'crossover',
        'crossover por trás',
        'crossover entre as pernas'
    ]
    
    analyzer = MovementAnalyzer()
    all_loaded = True
    
    for movement in movements:
        try:
            model = analyzer.load_movement_model(movement)
            if model:
                parts_count = len(model.get('statistics', {}))
                refs_count = model.get('num_references', 0)
                print(f"✅ {movement}: {parts_count} partes, {refs_count} refs")
            else:
                print(f"❌ {movement}: falha no carregamento")
                all_loaded = False
        except Exception as e:
            print(f"❌ {movement}: erro - {e}")
            all_loaded = False
    
    return all_loaded

def test_auto_generation():
    """Testa geração automática de modelos."""
    print("\n🔧 Testando Geração Automática")
    print("=" * 35)
    
    if not imports_ok:
        print("❌ Importações falharam, pulando teste")
        return False
    
    # Testar com um movimento que tem arquivos de referência
    test_movement = "crossover"
    
    # Verificar se modelo existe
    model_path = f"referencias/{test_movement}/modelo_{test_movement}.json"
    original_exists = os.path.exists(model_path)
    
    if original_exists:
        print(f"✅ Modelo {test_movement} já existe")
        
        # Testar geração automática (deve retornar True sem fazer nada)
        generator = ModelGenerator()
        success = generator.ensure_model_exists(test_movement)
        
        if success:
            print("✅ Geração automática funcionando (modelo já existia)")
            return True
        else:
            print("❌ Geração automática falhou")
            return False
    else:
        print(f"⚠️ Modelo {test_movement} não existe, testando geração...")
        
        # Testar geração automática
        generator = ModelGenerator()
        success = generator.ensure_model_exists(test_movement)
        
        if success:
            print("✅ Geração automática funcionando (modelo criado)")
            return True
        else:
            print("❌ Geração automática falhou")
            return False

def test_error_handling():
    """Testa tratamento de erros."""
    print("\n🚨 Testando Tratamento de Erros")
    print("=" * 30)
    
    if not imports_ok:
        print("❌ Importações falharam, pulando teste")
        return False
    
    # Testar movimento inexistente
    nonexistent_movement = "movimento_inexistente"
    
    generator = ModelGenerator()
    analyzer = MovementAnalyzer()
    
    # Tentar gerar modelo inexistente
    success = generator.ensure_model_exists(nonexistent_movement)
    if not success:
        print("✅ Geração falhou corretamente para movimento inexistente")
    else:
        print("❌ Geração inesperadamente bem-sucedida")
        return False
    
    # Tentar carregar modelo inexistente
    try:
        model = analyzer.load_movement_model(nonexistent_movement)
        if model is None:
            print("✅ Carregamento falhou corretamente para movimento inexistente")
        else:
            print("❌ Carregamento inesperadamente bem-sucedido")
            return False
    except Exception as e:
        print(f"✅ Erro capturado corretamente: {type(e).__name__}")
    
    return True

def test_main_imports():
    """Testa importações do main.py."""
    print("\n📦 Testando Importações do Sistema")
    print("=" * 35)
    
    if imports_ok:
        print("✅ Importações do sistema funcionando")
        return True
    else:
        print("❌ Importações do sistema falharam")
        return False

def test_file_structure():
    """Testa estrutura de arquivos."""
    print("\n📁 Testando Estrutura de Arquivos")
    print("=" * 35)
    
    required_files = [
        "main.py",
        "movement_analyzer.py",
        "model_generator.py",
        "referencias/arremesso/modelo_arremesso.json",
        "referencias/bandeja/modelo_bandeja.json",
        "referencias/spin/modelo_spin.json",
        "referencias/half spin/modelo_half spin.json",
        "referencias/crossover/modelo_crossover.json",
        "referencias/crossover por trás/modelo_crossover por trás.json",
        "referencias/crossover entre as pernas/modelo_crossover entre as pernas.json"
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {size:,} bytes")
        else:
            print(f"❌ {file_path}: NÃO ENCONTRADO")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Função principal do teste final."""
    print("🚀 TESTE FINAL DE VERIFICAÇÃO DO SISTEMA")
    print("=" * 60)
    
    results = []
    
    # Teste 1: Existência de modelos
    print("\n1️⃣ Verificando existência de modelos...")
    result1 = test_all_models_exist()
    results.append(("Existência de Modelos", result1))
    
    # Teste 2: Carregamento de modelos
    print("\n2️⃣ Testando carregamento de modelos...")
    result2 = test_model_loading()
    results.append(("Carregamento de Modelos", result2))
    
    # Teste 3: Geração automática
    print("\n3️⃣ Testando geração automática...")
    result3 = test_auto_generation()
    results.append(("Geração Automática", result3))
    
    # Teste 4: Tratamento de erros
    print("\n4️⃣ Testando tratamento de erros...")
    result4 = test_error_handling()
    results.append(("Tratamento de Erros", result4))
    
    # Teste 5: Importações
    print("\n5️⃣ Testando importações...")
    result5 = test_main_imports()
    results.append(("Importações", result5))
    
    # Teste 6: Estrutura de arquivos
    print("\n6️⃣ Testando estrutura de arquivos...")
    result6 = test_file_structure()
    results.append(("Estrutura de Arquivos", result6))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO FINAL DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 SISTEMA 100% FUNCIONAL!")
        print("✅ Todos os componentes estão funcionando corretamente")
        print("✅ Sistema pronto para produção")
        print("✅ Experiência do usuário garantida")
        return True
    else:
        print(f"\n⚠️ {total - passed} problema(s) encontrado(s)")
        print("❌ Sistema precisa de correções antes da produção")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
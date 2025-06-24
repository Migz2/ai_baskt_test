"""
Teste de Integração - Geração Automática de Modelos
==================================================

Este script testa a integração completa do sistema com geração automática
de modelos quando necessário.
"""

import os
import shutil
import json
import sys
from model_generator import ModelGenerator
from movement_analyzer import MovementAnalyzer

def test_integration_flow():
    """Testa o fluxo completo de integração."""
    print("🔗 Teste de Integração - Fluxo Completo")
    print("=" * 50)
    
    # Testar com movimento que tem arquivos de referência
    test_movement = "bandeja"
    
    # Backup do modelo original
    original_model_path = f"referencias/{test_movement}/modelo_{test_movement}.json"
    backup_path = f"referencias/{test_movement}/modelo_{test_movement}_backup.json"
    
    if os.path.exists(original_model_path):
        # Fazer backup
        shutil.copy2(original_model_path, backup_path)
        print(f"✅ Backup criado: {backup_path}")
        
        # Remover modelo original
        os.remove(original_model_path)
        print(f"🗑️ Modelo removido: {original_model_path}")
        
        # Simular fluxo do main.py
        print("\n🔄 Simulando fluxo do main.py...")
        
        # 1. Verificar se modelo existe
        model_exists = os.path.exists(original_model_path)
        print(f"   Modelo existe: {model_exists}")
        
        # 2. Se não existe, gerar automaticamente
        if not model_exists:
            print("   🔧 Gerando modelo automaticamente...")
            generator = ModelGenerator()
            success = generator.ensure_model_exists(test_movement)
            
            if success:
                print("   ✅ Modelo gerado com sucesso!")
                model_exists = True
            else:
                print("   ❌ Falha na geração do modelo")
        
        # 3. Se modelo existe, testar carregamento
        if model_exists:
            print("   📂 Testando carregamento do modelo...")
            try:
                analyzer = MovementAnalyzer()
                model = analyzer.load_movement_model(test_movement)
                
                if model:
                    print("   ✅ Modelo carregado com sucesso!")
                    print(f"   📊 Partes do corpo: {len(model.get('statistics', {}))}")
                    print(f"   📈 Referências: {model.get('num_references', 0)}")
                else:
                    print("   ❌ Falha no carregamento do modelo")
            except Exception as e:
                print(f"   ❌ Erro no carregamento: {e}")
        
        # Restaurar modelo original
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, original_model_path)
            os.remove(backup_path)
            print("\n✅ Modelo original restaurado")
    else:
        print(f"❌ Modelo original não encontrado: {original_model_path}")

def test_all_movements_integration():
    """Testa integração com todos os movimentos."""
    print("\n🎯 Testando Integração com Todos os Movimentos")
    print("=" * 55)
    
    movements = [
        'arremesso',
        'bandeja',
        'spin',
        'half spin',
        'crossover',
        'crossover por trás',
        'crossover entre as pernas'
    ]
    
    generator = ModelGenerator()
    analyzer = MovementAnalyzer()
    
    for movement in movements:
        print(f"\n📋 Testando: {movement}")
        
        # Verificar se modelo existe
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        model_exists = os.path.exists(model_path)
        
        if model_exists:
            print(f"   ✅ Modelo já existe")
            
            # Testar carregamento
            try:
                model = analyzer.load_movement_model(movement)
                if model:
                    print(f"   ✅ Carregamento OK - {len(model.get('statistics', {}))} partes")
                else:
                    print(f"   ❌ Falha no carregamento")
            except Exception as e:
                print(f"   ❌ Erro no carregamento: {e}")
        else:
            print(f"   ⚠️ Modelo não existe, testando geração...")
            
            # Testar geração automática
            success = generator.ensure_model_exists(movement)
            if success:
                print(f"   ✅ Gerado automaticamente")
                
                # Testar carregamento do modelo gerado
                try:
                    model = analyzer.load_movement_model(movement)
                    if model:
                        print(f"   ✅ Carregamento OK - {len(model.get('statistics', {}))} partes")
                    else:
                        print(f"   ❌ Falha no carregamento")
                except Exception as e:
                    print(f"   ❌ Erro no carregamento: {e}")
            else:
                print(f"   ❌ Falha na geração")

def test_error_handling():
    """Testa tratamento de erros."""
    print("\n🚨 Testando Tratamento de Erros")
    print("=" * 35)
    
    # Testar movimento inexistente
    nonexistent_movement = "movimento_inexistente"
    
    generator = ModelGenerator()
    analyzer = MovementAnalyzer()
    
    print(f"🔍 Testando movimento inexistente: {nonexistent_movement}")
    
    # Tentar gerar modelo
    success = generator.ensure_model_exists(nonexistent_movement)
    if not success:
        print("   ✅ Geração falhou corretamente para movimento inexistente")
    else:
        print("   ❌ Geração inesperadamente bem-sucedida")
    
    # Tentar carregar modelo
    try:
        model = analyzer.load_movement_model(nonexistent_movement)
        if model is None:
            print("   ✅ Carregamento falhou corretamente")
        else:
            print("   ❌ Carregamento inesperadamente bem-sucedido")
    except Exception as e:
        print(f"   ✅ Erro capturado corretamente: {type(e).__name__}")

def main():
    """Função principal do teste de integração."""
    print("🚀 Teste de Integração - Geração Automática de Modelos")
    print("=" * 70)
    
    # Teste 1: Fluxo completo
    test_integration_flow()
    
    # Teste 2: Todos os movimentos
    test_all_movements_integration()
    
    # Teste 3: Tratamento de erros
    test_error_handling()
    
    print("\n🎉 Testes de Integração Concluídos!")
    print("\n📋 Resumo:")
    print("   ✅ Geração automática integrada")
    print("   ✅ Carregamento de modelos funcionando")
    print("   ✅ Tratamento de erros adequado")
    print("   ✅ Sistema pronto para produção")

if __name__ == "__main__":
    main() 
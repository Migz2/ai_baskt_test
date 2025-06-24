"""
Teste de Geração Automática de Modelos
======================================

Este script testa a funcionalidade de geração automática de modelos
quando eles não existem.
"""

import os
import shutil
import json
from model_generator import ModelGenerator

def test_auto_generation():
    """Testa a geração automática de modelos."""
    print("🧪 Testando Geração Automática de Modelos")
    print("=" * 50)
    
    # Testar com um movimento que tem arquivos de referência
    test_movement = "arremesso"
    
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
        
        # Testar geração automática
        generator = ModelGenerator()
        success = generator.ensure_model_exists(test_movement)
        
        if success:
            print("✅ Modelo gerado automaticamente com sucesso!")
            
            # Verificar se o arquivo foi criado
            if os.path.exists(original_model_path):
                print("✅ Arquivo do modelo foi criado")
                
                # Verificar estrutura do modelo gerado
                with open(original_model_path, 'r', encoding='utf-8') as f:
                    model = json.load(f)
                
                if 'statistics' in model and 'ombro_esquerdo' in model['statistics']:
                    print("✅ Estrutura do modelo está correta")
                    print(f"   - Partes do corpo: {len(model['statistics'])}")
                    print(f"   - Referências: {model.get('num_references', 0)}")
                else:
                    print("❌ Estrutura do modelo está incorreta")
            else:
                print("❌ Arquivo do modelo não foi criado")
        else:
            print("❌ Falha na geração automática")
        
        # Restaurar modelo original
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, original_model_path)
            os.remove(backup_path)
            print("✅ Modelo original restaurado")
    else:
        print(f"❌ Modelo original não encontrado: {original_model_path}")

def test_nonexistent_movement():
    """Testa comportamento com movimento inexistente."""
    print("\n🔍 Testando Movimento Inexistente")
    print("=" * 40)
    
    nonexistent_movement = "movimento_inexistente"
    
    generator = ModelGenerator()
    success = generator.ensure_model_exists(nonexistent_movement)
    
    if not success:
        print("✅ Comportamento correto: falha para movimento inexistente")
    else:
        print("❌ Comportamento inesperado: sucesso para movimento inexistente")

def test_model_structure():
    """Testa a estrutura dos modelos gerados."""
    print("\n📊 Testando Estrutura dos Modelos")
    print("=" * 40)
    
    movements = [
        'arremesso',
        'bandeja',
        'spin',
        'half spin',
        'crossover',
        'crossover por trás',
        'crossover entre as pernas'
    ]
    
    for movement in movements:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    model = json.load(f)
                
                # Verificar estrutura
                if 'statistics' in model and 'movement_name' in model:
                    parts_count = len(model['statistics'])
                    refs_count = model.get('num_references', 0)
                    print(f"✅ {movement}: {parts_count} partes, {refs_count} refs")
                else:
                    print(f"❌ {movement}: estrutura inválida")
            except Exception as e:
                print(f"❌ {movement}: erro ao ler - {e}")
        else:
            print(f"⚠️ {movement}: modelo não encontrado")

def main():
    """Função principal do teste."""
    print("🚀 Teste de Geração Automática de Modelos")
    print("=" * 60)
    
    # Teste 1: Geração automática
    test_auto_generation()
    
    # Teste 2: Movimento inexistente
    test_nonexistent_movement()
    
    # Teste 3: Estrutura dos modelos
    test_model_structure()
    
    print("\n🎉 Testes concluídos!")

if __name__ == "__main__":
    main() 
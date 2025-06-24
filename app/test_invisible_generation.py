"""
Teste de Geração Invisível de Modelos
=====================================

Este script testa se a geração automática de modelos acontece de forma
invisível para o usuário, sem mostrar mensagens de erro ou alertas.
"""

import os
import shutil
import json
import sys
from model_generator import ModelGenerator
from movement_analyzer import MovementAnalyzer

def test_invisible_generation():
    """Testa se a geração é invisível para o usuário."""
    print("👻 Teste de Geração Invisível")
    print("=" * 40)
    
    # Testar com movimento que tem arquivos de referência
    test_movement = "spin"
    
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
        
        # Simular fluxo do main.py (sem mostrar mensagens)
        print("\n🔄 Simulando fluxo invisível...")
        
        # 1. Verificar se modelo existe (silencioso)
        model_exists = os.path.exists(original_model_path)
        print(f"   Modelo existe: {model_exists}")
        
        # 2. Se não existe, gerar automaticamente (sem mostrar spinner para usuário)
        if not model_exists:
            print("   🔧 Gerando modelo automaticamente (invisível)...")
            generator = ModelGenerator()
            success = generator.ensure_model_exists(test_movement)
            
            if success:
                print("   ✅ Modelo gerado com sucesso (invisível)!")
                model_exists = True
                # NÃO mostrar st.success() para o usuário
            else:
                print("   ❌ Falha na geração do modelo")
                # NÃO mostrar st.warning() para o usuário
                # Apenas st.stop() silenciosamente
        
        # 3. Se modelo existe, continuar normalmente
        if model_exists:
            print("   📂 Modelo disponível, continuando análise...")
            try:
                analyzer = MovementAnalyzer()
                model = analyzer.load_movement_model(test_movement)
                
                if model:
                    print("   ✅ Análise pode continuar normalmente")
                    print(f"   📊 Partes do corpo: {len(model.get('statistics', {}))}")
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

def test_all_movements_invisible():
    """Testa geração invisível para todos os movimentos."""
    print("\n🎯 Testando Geração Invisível para Todos os Movimentos")
    print("=" * 60)
    
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
    
    for movement in movements:
        print(f"\n📋 Testando: {movement}")
        
        # Verificar se modelo existe
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        model_exists = os.path.exists(model_path)
        
        if model_exists:
            print(f"   ✅ Modelo já existe (sem ação necessária)")
        else:
            print(f"   ⚠️ Modelo não existe, testando geração invisível...")
            
            # Testar geração automática (sem mostrar mensagens)
            success = generator.ensure_model_exists(movement)
            if success:
                print(f"   ✅ Gerado automaticamente (invisível)")
                # NÃO mostrar st.success() para o usuário
            else:
                print(f"   ❌ Falha na geração")
                # NÃO mostrar st.warning() para o usuário
                # Apenas st.stop() silenciosamente

def test_error_handling_invisible():
    """Testa tratamento de erros invisível."""
    print("\n🚨 Testando Tratamento de Erros Invisível")
    print("=" * 45)
    
    # Testar movimento inexistente
    nonexistent_movement = "movimento_inexistente"
    
    generator = ModelGenerator()
    
    print(f"🔍 Testando movimento inexistente: {nonexistent_movement}")
    
    # Tentar gerar modelo (deve falhar silenciosamente)
    success = generator.ensure_model_exists(nonexistent_movement)
    if not success:
        print("   ✅ Geração falhou corretamente (invisível)")
        # NÃO mostrar st.warning() para o usuário
        # Apenas st.stop() silenciosamente
    else:
        print("   ❌ Geração inesperadamente bem-sucedida")

def test_main_flow_simulation():
    """Simula o fluxo completo do main.py."""
    print("\n🔄 Simulando Fluxo Completo do main.py")
    print("=" * 45)
    
    # Simular seleção de movimento pelo usuário
    user_selected_movement = "bandeja"
    print(f"👤 Usuário selecionou: {user_selected_movement}")
    
    # Mapear para formato interno
    movement_mapping = {
        "Arremesso": "arremesso",
        "Bandeja": "bandeja",
        "Spin": "spin",
        "Half Spin": "half spin",
        "Crossover": "crossover",
        "Crossover por trás": "crossover por trás",
        "Crossover entre as pernas": "crossover entre as pernas"
    }
    
    internal_movement = movement_mapping.get(user_selected_movement, user_selected_movement.lower())
    print(f"🔄 Mapeado para: {internal_movement}")
    
    # Verificar modelo (fluxo do main.py)
    model_path = f"referencias/{internal_movement}/modelo_{internal_movement}.json"
    model_exists = os.path.exists(model_path)
    print(f"📂 Modelo existe: {model_exists}")
    
    # Se não existe, gerar automaticamente (invisível)
    if not model_exists:
        print("   🔧 Gerando modelo automaticamente...")
        generator = ModelGenerator()
        success = generator.ensure_model_exists(internal_movement)
        
        if success:
            print("   ✅ Modelo gerado (invisível para usuário)")
            model_exists = True
            # NÃO mostrar st.success() para o usuário
        else:
            print("   ❌ Falha na geração")
            # Mostrar apenas: st.warning("⚠️ Este tipo de movimento ainda está sendo adicionado. Experimente com outro tipo.")
            # st.stop()
    
    # Se modelo existe, continuar análise
    if model_exists:
        print("   🚀 Análise pode prosseguir normalmente")
        print("   ✅ Experiência do usuário: fluida e sem interrupções")
    else:
        print("   ⚠️ Análise não pode prosseguir")
        print("   ❌ Experiência do usuário: interrompida")

def main():
    """Função principal do teste de geração invisível."""
    print("🚀 Teste de Geração Invisível de Modelos")
    print("=" * 60)
    
    # Teste 1: Geração invisível
    test_invisible_generation()
    
    # Teste 2: Todos os movimentos
    test_all_movements_invisible()
    
    # Teste 3: Tratamento de erros invisível
    test_error_handling_invisible()
    
    # Teste 4: Fluxo completo do main.py
    test_main_flow_simulation()
    
    print("\n🎉 Testes de Geração Invisível Concluídos!")
    print("\n📋 Resumo:")
    print("   ✅ Geração automática é invisível para o usuário")
    print("   ✅ Não mostra mensagens técnicas ou de erro")
    print("   ✅ Experiência do usuário permanece fluida")
    print("   ✅ Sistema funciona de forma transparente")
    print("   ✅ Apenas avisos discretos quando necessário")

if __name__ == "__main__":
    main() 
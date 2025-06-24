"""
Script para Garantir Modelos de Referência
==========================================

Este script verifica e cria automaticamente todos os modelos de referência
necessários para o funcionamento do sistema.
"""

import os
import sys
from reference_processor import ReferenceProcessor, process_all_movements

def ensure_all_models():
    """Garante que todos os modelos de referência existam."""
    print("🔧 Verificando e criando modelos de referência...")
    
    # Lista de movimentos que devem ter modelos
    required_movements = [
        "arremesso",
        "bandeja", 
        "spin",
        "half spin",
        "crossover",
        "crossover por trás",
        "crossover entre as pernas"
    ]
    
    # Verificar quais modelos já existem
    existing_models = []
    missing_models = []
    
    for movement in required_movements:
        model_path = f"referencias/{movement}/modelo_{movement}.json"
        if os.path.exists(model_path):
            existing_models.append(movement)
            print(f"✅ Modelo existente: {movement}")
        else:
            missing_models.append(movement)
            print(f"❌ Modelo ausente: {movement}")
    
    # Se há modelos ausentes, criar todos
    if missing_models:
        print(f"\n🔄 Criando {len(missing_models)} modelos ausentes...")
        
        try:
            # Processar todos os movimentos
            results = process_all_movements()
            
            if results:
                print(f"✅ {len(results)} modelos criados com sucesso!")
                
                # Verificar novamente
                for movement in required_movements:
                    model_path = f"referencias/{movement}/modelo_{movement}.json"
                    if os.path.exists(model_path):
                        print(f"✅ Modelo confirmado: {movement}")
                    else:
                        print(f"❌ Falha ao criar: {movement}")
            else:
                print("❌ Falha ao criar modelos")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante criação dos modelos: {e}")
            return False
    else:
        print(f"\n🎉 Todos os {len(existing_models)} modelos já existem!")
    
    return True

def verify_model_structure():
    """Verifica se a estrutura dos modelos está correta."""
    print("\n🔍 Verificando estrutura dos modelos...")
    
    required_movements = [
        "arremesso",
        "bandeja", 
        "spin",
        "half spin",
        "crossover",
        "crossover por trás",
        "crossover entre as pernas"
    ]
    
    processor = ReferenceProcessor()
    
    for movement in required_movements:
        model = processor.load_movement_model(movement)
        if model and 'statistics' in model:
            parts_count = len(model['statistics'])
            print(f"✅ {movement}: {parts_count} partes do corpo")
        else:
            print(f"❌ {movement}: estrutura inválida")
    
    processor.close()

def main():
    """Função principal."""
    print("🚀 Iniciando verificação de modelos de referência")
    print("=" * 60)
    
    # Garantir que todos os modelos existam
    success = ensure_all_models()
    
    if success:
        # Verificar estrutura
        verify_model_structure()
        print("\n🎉 Sistema pronto para uso!")
    else:
        print("\n❌ Falha na preparação dos modelos")
        sys.exit(1)

if __name__ == "__main__":
    main() 
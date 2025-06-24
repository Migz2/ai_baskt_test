"""
Exemplo de Uso do Sistema de Processamento de Referências
=========================================================

Este script demonstra como usar o ReferenceProcessor para:
1. Processar um movimento específico
2. Processar todos os movimentos
3. Carregar e usar modelos existentes
4. Verificar estatísticas dos movimentos

EXECUÇÃO:
python example_processing.py
"""

import os
import json
from reference_processor import ReferenceProcessor, process_all_movements

def exemplo_processamento_individual():
    """Exemplo 1: Processar um movimento específico."""
    print("\n" + "="*60)
    print("📹 EXEMPLO 1: Processamento Individual")
    print("="*60)
    
    # Criar processador
    processor = ReferenceProcessor()
    
    try:
        # Processar movimento "arremesso"
        print("🔄 Processando movimento 'arremesso'...")
        model = processor.process_movement_references("arremesso")
        
        if model:
            print("✅ Processamento concluído!")
            print(f"   📊 Referências processadas: {model['num_references']}")
            print(f"   🎬 Frames totais: {model['total_frames']}")
            print(f"   🏃 Partes do corpo: {len(model['body_parts'])}")
            
            # Mostrar exemplo de estatísticas
            first_part = list(model['statistics'].keys())[0]
            print(f"\n📈 Exemplo de estatísticas ({first_part}):")
            stats = model['statistics'][first_part]
            
            print(f"   Keypoints - Média: {stats['keypoints']['mean'][:2]}...")  # Primeiros 2 valores
            print(f"   Keypoints - Desvio: {stats['keypoints']['std'][:2]}...")
            print(f"   Vetores - Média: {stats['vectors']['mean'][:2]}...")
            print(f"   Vetores - Desvio: {stats['vectors']['std'][:2]}...")
        else:
            print("❌ Falha no processamento")
    
    finally:
        processor.close()

def exemplo_processamento_lote():
    """Exemplo 2: Processar todos os movimentos."""
    print("\n" + "="*60)
    print("🚀 EXEMPLO 2: Processamento em Lote")
    print("="*60)
    
    print("🔄 Processando todos os movimentos disponíveis...")
    results = process_all_movements()
    
    if results:
        print(f"✅ {len(results)} movimentos processados!")
        
        for movement, model in results.items():
            print(f"\n📊 {movement.upper()}:")
            print(f"   Referências: {model['num_references']}")
            print(f"   Frames: {model['total_frames']}")
            print(f"   Modelo salvo: referencias/{movement}/modelo_{movement}.json")
    else:
        print("❌ Falha no processamento em lote")

def exemplo_carregamento_modelo():
    """Exemplo 3: Carregar e usar modelo existente."""
    print("\n" + "="*60)
    print("📂 EXEMPLO 3: Carregamento de Modelo")
    print("="*60)
    
    processor = ReferenceProcessor()
    
    try:
        # Carregar modelo de bandeja
        print("📂 Carregando modelo 'bandeja'...")
        model = processor.load_movement_model("bandeja")
        
        if model:
            print("✅ Modelo carregado com sucesso!")
            print(f"   📊 Nome: {model['movement_name']}")
            print(f"   🎬 Referências: {model['num_references']}")
            print(f"   🏃 Partes do corpo: {len(model['body_parts'])}")
            print(f"   📅 Criado em: {model['created_at']}")
            
            # Mostrar todas as partes do corpo
            print(f"\n🏃 Partes do corpo analisadas:")
            for i, part in enumerate(model['body_parts'], 1):
                print(f"   {i:2d}. {part}")
            
            # Mostrar estatísticas detalhadas de uma parte
            print(f"\n📈 Estatísticas detalhadas (ombro_esquerdo):")
            stats = model['statistics']['ombro_esquerdo']
            
            print(f"   Keypoints:")
            print(f"     Média: {stats['keypoints']['mean']}")
            print(f"     Desvio: {stats['keypoints']['std']}")
            print(f"     Mínimo: {stats['keypoints']['min']}")
            print(f"     Máximo: {stats['keypoints']['max']}")
            
            print(f"   Vetores:")
            print(f"     Média: {stats['vectors']['mean']}")
            print(f"     Desvio: {stats['vectors']['std']}")
            print(f"     Mínimo: {stats['vectors']['min']}")
            print(f"     Máximo: {stats['vectors']['max']}")
        else:
            print("❌ Modelo não encontrado")
    
    finally:
        processor.close()

def exemplo_verificacao_arquivos():
    """Exemplo 4: Verificar arquivos gerados."""
    print("\n" + "="*60)
    print("📁 EXEMPLO 4: Verificação de Arquivos")
    print("="*60)
    
    # Verificar estrutura de arquivos
    referencias_dir = "referencias"
    
    if not os.path.exists(referencias_dir):
        print("❌ Pasta de referências não encontrada")
        return
    
    print(f"📂 Pasta de referências: {referencias_dir}")
    
    # Listar movimentos processados
    movements = [d for d in os.listdir(referencias_dir) 
                if os.path.isdir(os.path.join(referencias_dir, d))]
    
    if not movements:
        print("❌ Nenhum movimento processado encontrado")
        return
    
    print(f"✅ Movimentos encontrados: {len(movements)}")
    
    for movement in movements:
        movement_dir = os.path.join(referencias_dir, movement)
        files = os.listdir(movement_dir)
        
        # Contar tipos de arquivo
        keypoints_files = [f for f in files if f.endswith('_keypoints.json')]
        vectors_files = [f for f in files if f.endswith('_vetores.json')]
        model_files = [f for f in files if f.startswith('modelo_')]
        
        print(f"\n📊 {movement.upper()}:")
        print(f"   Keypoints: {len(keypoints_files)} arquivos")
        print(f"   Vetores: {len(vectors_files)} arquivos")
        print(f"   Modelos: {len(model_files)} arquivos")
        
        # Mostrar tamanho do modelo
        if model_files:
            model_file = os.path.join(movement_dir, model_files[0])
            size_kb = os.path.getsize(model_file) / 1024
            print(f"   Tamanho do modelo: {size_kb:.1f} KB")

def exemplo_estatisticas_comparativas():
    """Exemplo 5: Comparar estatísticas entre movimentos."""
    print("\n" + "="*60)
    print("📊 EXEMPLO 5: Estatísticas Comparativas")
    print("="*60)
    
    processor = ReferenceProcessor()
    
    try:
        # Carregar modelos disponíveis
        movements = ["bandeja", "arremesso"]
        models = {}
        
        for movement in movements:
            model = processor.load_movement_model(movement)
            if model:
                models[movement] = model
        
        if not models:
            print("❌ Nenhum modelo encontrado para comparação")
            return
        
        print(f"✅ Modelos carregados: {len(models)}")
        
        # Comparar estatísticas
        print(f"\n📈 Comparação de Estatísticas:")
        print(f"{'Movimento':<15} {'Refs':<5} {'Frames':<8} {'Partes':<8}")
        print("-" * 40)
        
        for movement, model in models.items():
            print(f"{movement:<15} {model['num_references']:<5} "
                  f"{model['total_frames']:<8} {len(model['body_parts']):<8}")
        
        # Comparar uma parte específica
        if len(models) >= 2:
            print(f"\n🔍 Comparação detalhada (ombro_esquerdo):")
            part = "ombro_esquerdo"
            
            for movement, model in models.items():
                stats = model['statistics'][part]
                mean_keypoints = stats['keypoints']['mean']
                mean_vectors = stats['vectors']['mean']
                
                print(f"\n{movement.upper()}:")
                print(f"   Keypoints (média): [{mean_keypoints[0]:.3f}, {mean_keypoints[1]:.3f}, {mean_keypoints[2]:.3f}]")
                print(f"   Vetores (média): [{mean_vectors[0]:.3f}, {mean_vectors[1]:.3f}, {mean_vectors[2]:.3f}]")
    
    finally:
        processor.close()

def main():
    """Executar todos os exemplos."""
    print("🎯 EXEMPLOS DE USO DO SISTEMA DE PROCESSAMENTO")
    print("=" * 60)
    
    # Executar exemplos
    exemplo_processamento_individual()
    exemplo_processamento_lote()
    exemplo_carregamento_modelo()
    exemplo_verificacao_arquivos()
    exemplo_estatisticas_comparativas()
    
    print("\n" + "="*60)
    print("✅ TODOS OS EXEMPLOS CONCLUÍDOS!")
    print("="*60)
    print("\n📝 Próximos passos:")
    print("1. Use o sistema no Streamlit: streamlit run main.py")
    print("2. Processe mais movimentos conforme necessário")
    print("3. Integre com o sistema de análise de usuários")
    print("4. Ajuste parâmetros baseado nos resultados")

if __name__ == "__main__":
    main() 
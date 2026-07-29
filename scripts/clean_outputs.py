import os
import shutil
from config import Config

def remove_dir(dir_path):
    if os.path.exists(dir_path):
        print(f"Limpando o conteúdo do diretório: {dir_path}")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Falha ao apagar {file_path}. Razão: {e}")
    else:
        print(f"Diretório não existe, ignorando: {dir_path}")

def clean_all_experiments():
    """
    Remove TODOS os checkpoints, logs e outputs gráficos anteriores 
    para garantir que os 4 testes comecem do zero de forma isolada.
    """
    print("Iniciando limpeza de experimentos antigos...")
    
    # Base paths
    base_checkpoints = os.path.join(Config.BASE_PATH, 'checkpoints')
    base_logs = os.path.join(Config.BASE_PATH, 'lightning_csv_logs')
    base_outputs = os.path.join(os.getcwd(), 'outputs')

    # Pergunta de confirmação de segurança
    print(f"Isso irá DELETAR as seguintes pastas (e todo o seu conteúdo):")
    print(f"1. {base_checkpoints}")
    print(f"2. {base_logs}")
    print(f"3. {base_outputs}")
    
    resp = input("Tem certeza que deseja apagar os arquivos antigos? (s/n): ")
    
    if resp.lower() == 's':
        remove_dir(base_checkpoints)
        remove_dir(base_logs)
        remove_dir(base_outputs)
        print("\nPronto! Tudo limpo. Você já pode iniciar o treinamento controlado alterando as flags no config.py.")
    else:
        print("\nOperação cancelada.")

if __name__ == "__main__":
    clean_all_experiments()

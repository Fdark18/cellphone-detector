import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("lakshyataragi/mobilephoneusagedatasetiitr")
print("Path original:", path)

# Pasta atual
pasta_atual = os.getcwd()

# Mover arquivos para pasta atual
for arquivo in os.listdir(path):
    origem = os.path.join(path, arquivo)
    destino = os.path.join(pasta_atual, arquivo)
    shutil.move(origem, destino)
    print(f"Movido: {arquivo}")

# Apagar pasta do cache
cache_path = r"C:\Users\jhony\.cache\kagglehub"
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print("Cache apagado!")

print("Arquivos na pasta atual:", os.listdir(pasta_atual))
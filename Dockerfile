# Usa uma imagem oficial do PyTorch com suporte a CUDA para rodar na GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala dependências de sistema exigidas pelo OpenCV (e.g. libGL)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de dependência e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os módulos do projeto para o contêiner
COPY . .

# Define o caminho do dataset (montado via volume no docker-compose)
ENV DATASET_PATH=/app/data

# Comando padrão ao iniciar o contêiner
CMD ["python", "train.py"]

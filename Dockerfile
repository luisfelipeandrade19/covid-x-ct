# Usa uma imagem oficial do PyTorch com suporte a CUDA para rodar na GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos de dependência e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os módulos do projeto para o contêiner
COPY config.py .
COPY dataset.py .
COPY loaders.py .
COPY model.py .
COPY train.py .
COPY evaluate.py .
COPY utils.py .
COPY callbacks.py .

# Configura uma variável de ambiente para o kagglehub salvar os downloads em uma pasta específica
ENV KAGGLEHUB_CACHE=/app/data

# Comando padrão ao iniciar o contêiner
CMD ["python", "train.py"]

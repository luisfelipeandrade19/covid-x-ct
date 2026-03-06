# Usa uma imagem oficial do PyTorch com suporte a CUDA para rodar na GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos de dependência e o script para o contêiner
COPY requirements.txt .
COPY covidxct.py .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Configura uma variável de ambiente para o kagglehub salvar os downloads em uma pasta específica
ENV KAGGLEHUB_CACHE=/app/data

# Comando padrão ao iniciar o contêiner
CMD ["python", "covidxct.py"]
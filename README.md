# COVID-X-CT

Um classificador de imagens de Tomografia Computadorizada (CT) usando Deep Learning para o diagnóstico de COVID-19, Pneumonia e casos Normais.

Este projeto utiliza o framework **PyTorch Lightning** e aplica transfer learning a partir de um modelo **DenseNet161** pré-treinado. O código baixa automaticamente o dataset original do Kaggle e gerencia o treinamento, a validação e a geração de uma matriz de confusão dos resultados.

## 🚀 Tecnologias

- **Python 3**
- **PyTorch** & **PyTorch Lightning**: Para criação e treinamento do modelo de Deep Learning.
- **OpenCV** e **Pillow**: Para processamento de imagem (como aplicação de redimensionamento, tons de cinza e o algoritmo de aprimoramento de contraste CLAHE).
- **Kagglehub**: Para download automático e extração dos datasets hospedados no Kaggle (`hgunraj/covidxct`).
- **Docker** & **Docker Compose**: Para construir e gerenciar de forma repetível a execução do código em containers isolados, com suporte a inferência acelerada por GPU NVIDIA.

## 📂 Visão Geral dos Arquivos

- `covidxct.py`: Script principal que faz o fluxo completo: donwload de dados, construção dos Datasets e DataLoaders, definição do modelo (SimpleClassifier usando Densenet161), treinamento com o `pl.Trainer`, teste no conjunto de validação e exibição das métricas e matriz de confusão final.
- `Dockerfile`: Configuração para a construção do container Docker usando uma imagem base oficial do PyTorch com suporte a CUDA 11.8. 
- `docker-compose.yml`: Manifesto simplificado para orquestrar o container, injetar volume local para os datasets baixados não serem perdidos e solicitar acesso à GPU nvidia.
- `requirements.txt`: Todas as bibliotecas Python necessárias para que o script execute sem problemas, instaladas automaticamente pelo Dockerfile.

## 🛠 Como rodar localmente com Docker (Recomendado)

A melhor forma de rodar o projeto de forma padronizada e usando GPU é utilizar o **Docker** em conjunto com **Docker Compose** e o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### 1. Pré-Requisitos
- Docker e Docker Compose devidamente instalados.
- Placa de Vídeo compatível NVIDIA com os devidos drivers de versão atualizada.
- Instalação e configuração do NVIDIA Container Runtime para que o docker tenha acesso nativo à GPU.

### 2. Configurando os volumes do dataset

No arquivo `docker-compose.yml`, note que a pasta em cache local para os arquivos do Kaggle aponta para `D:\Dataset`. Edite este arquivo se desejar mudar o destino onde os 3GB+ de dados de tomografia serão salvos na sua máquina para não precisar fazer o download toda a vez:
```yaml
    volumes:
      - D:\Dataset:/app/data
```

### 3. Rodando o projeto

Abra o terminal na pasta raiz deste projeto e execute:
```bash
docker-compose up --build
```
A imagem docker será baixada e criada. Em seguida, os dados do Kaggle serão obtidos. O modelo começará o treinamento iterativo. No final do processo (ao completar 10 épocas ou estourar as limitações fornecidas), ele avaliará o modelo de validação, irá printar o Relatório de Classificação e por fim criará e exibirá a Matriz de Confusão em `matriz-confusao.png`.

## 🖥 Otimizações Aplicadas e Detalhes de Código

1. **Deadlocks**: A leitura de dados em dataloaders do PyTorch usando OpenCV num ambiente de Docker ou Kaggle pode ser bloqueada quando várias `num_workers` são instanciadas. Aqui o script força `cv2.setNumThreads(0)` e `num_workers=0`.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Antes de serem enviadas para o modelo, cada imagem de raio X é processada matematicamente pelo algoritmo CLAHE para intensificar bordas e diferenças nos tecidos pulmonares de maneira controlada.
3. **Fine-tuning Parcial (Transfer Learning)**: Toda a arquitetura do modelo DenseNet161 e seus blocos convulsionais são intencionalmente "congelados" (pesos imutáveis) exceto no bloco final e camandas de normalização recentes (`'denseblock4'` e `'norm5'`). O head linear também é substituído por uma `nn.Linear` que enuncia somente para o número condicionado de saídas (`Config.NUM_CLASSES = 3`, ou Normal, Pneumonia e Covid-19). Essa técnica economiza VRAM substancial, estabilidade, mas permite refinar e adaptar a rede para padrões sutis em CT de tórax.

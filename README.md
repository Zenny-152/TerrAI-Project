### TerrAI — Detecção de Risco de Deslizamentos com IA

O TerrAI é um sistema desenvolvido como Trabalho de Conclusão de Curso (TCC) com o objetivo de auxiliar na identificação de risco de deslizamentos de encostas a partir de imagens capturadas por usuários.

A proposta do projeto é permitir que qualquer pessoa envie uma foto de um terreno e receba uma estimativa automática de risco utilizando técnicas de **Visão Computacional** e **Aprendizado de Máquina**.

---

## Funcionalidades

-  Upload de imagens via interface web
-  Classificação automática de risco com modelo de IA
-  Retorno de:
  - Classe de risco
  - Probabilidade da predição
  - Score contínuo de risco
-  API REST para integração com outros sistemas
-  Leitura opcional de geolocalização (EXIF)

---

## Tecnologias Utilizadas

### Machine Learning
- Python
- PyTorch
- Torchvision
- Transfer Learning (ResNet)
- Computer Vision

### Backend
- Flask (API REST)
- Python
- Logging

### Frontend
- HTML5
- CSS3
- JavaScript (Vanilla)
- EXIF.js

### Banco de Dados
- PostgreSQL

---

## Arquitetura do Projeto

O sistema segue uma arquitetura modular dividida em três partes principais:

[Frontend]
↓
[API Flask]
↓
[Módulo de Machine Learning]

---

Fluxo:
1. Usuário envia uma imagem
2. Backend recebe e processa o arquivo
3. Modelo de IA realiza a inferência
4. Resultado é retornado para o frontend

---

##  Modelo de Machine Learning

O modelo foi desenvolvido utilizando **Transfer Learning** com redes pré-treinadas no ImageNet.

- Arquitetura: ResNet
- Classes:
  - Low (baixo risco)
  - Medium (médio risco)
  - High (alto risco)
- Saída:
  - Probabilidades por classe (Softmax)
  - Classe final
  - Score de risco

---

##  Avaliação

O modelo foi avaliado com métricas como:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusão

> ⚠️ Observação: Este projeto é uma prova de conceito e possui limitações relacionadas ao tamanho e qualidade do dataset.

---

## Como executar o projeto

### 1. Clonar o repositório
git clone https://github.com/seu-usuario/terrai.git
cd terrai

### 2. Criar ambiente virtual

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

### 3. Instalar dependências

pip install -r requirements.txt

### 4. Configurar variáveis de ambiente

Crien um arquivo .env com:
ML_MODEL_PATH=ml/models/best.pth

### 5. Executar aplicação

python main.py

### A aplicação estará disponível em:

http://localhost:5000

---

### Estrutura do projeto

```
terrai/
│
├── backend/
├── ml/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── templates/
├── static/
├── main.py
└── requirements.txt
```

---

### Trabalhos Futuros

Integração com dados geoespaciais (PostGIS)
Aumento do dataset
Melhoria da acurácia do modelo
Deploy em ambiente cloud
Aplicação mobile

---

### Autor

Lucas Muniz Vieira Souto

TerrAI — Detecção de Risco de Deslizamentos com IA

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

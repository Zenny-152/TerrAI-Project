# ml/model.py
"""
Modelo placeholder: aqui ficará a definição do modelo real.
Para o protótipo do TCC usamos algo simples (por exemplo: classificação binária com transfer-learning)
"""

from typing import Any

class FakeModel:
    def __init__(self):
        # aqui carregaríamos pesos, etc.
        self.name = "fake-model-v0"

    def predict_proba(self, image_bytes: bytes) -> float:
        """
        Recebe bytes da imagem e retorna probabilidade de deslizamento (0.0 - 1.0).
        Placeholder: retorna uma pontuação determinística simples a partir do hash dos bytes,
        assim os resultados não são totalmente aleatórios entre requisições.
        """
        # cálculo simples e estável (não aleatório): hash -> float
        h = sum(image_bytes) if image_bytes else 0
        # normaliza para 0..1
        prob = ((h % 1000) / 1000.0)
        return float(prob)

# única instância global do "modelo"
MODEL = FakeModel()

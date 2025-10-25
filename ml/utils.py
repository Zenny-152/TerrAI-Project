from typing import List

CLASS_NAMES = ["01_low", "02_medium", "03_high", "04_extreme"]

# mapeamento human-friendly (opcional)
CLASS_DISPLAY_NAMES = {
    "01_low": "low",
    "02_medium": "medium",
    "03_high": "high",
    "04_extreme": "extreme"
}

# centers used to convert class probs into a representative percentage
CLASS_CENTERS = [15.0, 45.5, 75.5, 95.0]

def probs_to_percentage(probs: List[float]) -> float:
    """
    Recebe uma lista/np.array de probabilidades (softmax) e calcula uma
    porcentagem representativa usando os centros.
    """
    if not probs:
        return 0.0
    # garante mesmo tamanho
    n = min(len(probs), len(CLASS_CENTERS))
    s = 0.0
    for i in range(n):
        s += float(probs[i]) * float(CLASS_CENTERS[i])
    return round(s, 2)

def percentage_to_bucket(percent: float) -> str:
    """
    Converte a porcentagem para rótulo textual conforme suas faixas:
      0..30    -> low
      31..60   -> medium
      61..90   -> high
      >90     -> extreme
    """
    try:
        percent = float(percent)
    except Exception:
        return "unknown"
    if percent <= 30:
        return "low"
    if percent <= 60:
        return "medium"
    if percent <= 90:
        return "high"
    return "extreme"

def class_index_to_name(idx: int) -> str:
    try:
        return CLASS_NAMES[int(idx)]
    except Exception:
        return "unknown"

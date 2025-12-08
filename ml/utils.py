from typing import List

CLASS_NAMES = ["01_low", "02_medium", "03_high"]

# mapeamento human-friendly
CLASS_DISPLAY_NAMES = {
    "01_low": "low",
    "02_medium": "medium",
    "03_high": "high",
}

# centers used to convert class probs into a representative percentage
CLASS_CENTERS = {
    "01_low": 15.0,     # centro de 0-30
    "02_medium": 48.0,  # centro de 31-65 (aprox 48)
    "03_high": 85.0     # centro de 66-100
}

def probs_to_percentage(probs: List[float]) -> float:
    """
    Recebe uma lista/np.array de probabilidades (softmax) e calcula uma
    porcentagem representativa usando os centros (CLASS_CENTERS) mapeados
    na ordem de CLASS_NAMES.
    """
    if not probs:
        return 0.0
    # usa a ordem explícita de CLASS_NAMES para referenciar os centros
    s = 0.0
    n = min(len(probs), len(CLASS_NAMES))
    for i in range(n):
        class_key = CLASS_NAMES[i]  # ex: "01_low"
        center = CLASS_CENTERS.get(class_key)
        if center is None:
            # fallback: evenly spaced center aproximado (por segurança)
            center = (i + 0.5) * (100.0 / len(CLASS_NAMES))
        s += float(probs[i]) * float(center)
    return round(s, 2)

def percentage_to_bucket(percent: float) -> str:
    """
    Converte a porcentagem para rótulo textual conforme suas faixas:
      0..30    -> low
      31..65   -> medium
      66..100   -> high
    """
    try:
        percent = float(percent)
    except Exception:
        return "unknown"
    if percent <= 30:
        return "low"
    if percent <= 65:
        return "medium"
    return "high"

def class_index_to_name(idx: int) -> str:
    try:
        return CLASS_NAMES[int(idx)]
    except Exception:
        return "unknown"

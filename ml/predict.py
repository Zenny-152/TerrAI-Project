import io
import os
import logging
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from .utils import CLASS_NAMES, CLASS_DISPLAY_NAMES, percentage_to_bucket, probs_to_percentage

# Configs
MODEL_PATH = os.getenv("ML_MODEL_PATH", "ml/models/best.pth")  # ajuste se necessário
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = len(CLASS_NAMES)
logger = logging.getLogger(__name__)

# transforms (mesmos do treino/val)
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

_model = None

def _build_model(num_classes=NUM_CLASSES):
    # mesma arquitetura do treino (ResNet18) — adapta se treinou outra
    model = models.resnet18(weights=None)
    num_f = model.fc.in_features
    model.fc = torch.nn.Linear(num_f, num_classes)
    return model

def load_model(path: str = None):
    global _model, CLASS_NAMES
    if _model is not None:
        return _model
    p = path or MODEL_PATH
    device = DEVICE
    model = _build_model(NUM_CLASSES)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}")
    state = torch.load(p, map_location=device)
    ckpt_class_to_idx = None
    if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
        sd = state.get("state_dict") or state.get("model_state_dict")
        class_to_idx = state.get("class_to_idx") or state.get("class_mapping")
        if class_to_idx:
            try:
                # invert mapping: idx -> class_name
                inv = {int(v): str(k) for k, v in class_to_idx.items()}
                # Create ordered class list based on indices 0..n-1
                max_idx = max(inv.keys())
                ordered = [inv[i] for i in range(max_idx + 1)]
                CLASS_NAMES = ordered
                logger.info("Loaded class_to_idx from checkpoint; CLASS_NAMES set to: %s", CLASS_NAMES)
                ckpt_class_to_idx = class_to_idx
            except Exception as e:
                logger.debug("Failed to apply class_to_idx from checkpoint: %s", e)
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model = model
    # Return model and any class map found via attribute for debug if needed
    _model._ckpt_class_to_idx = ckpt_class_to_idx
    return _model

def _prepare_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0)

def _normalize_percent_value(val):
    """Garantir percent em 0..100 (float). Aceita 0..1 ou 0..100."""
    try:
        v = float(val)
    except Exception:
        return None
    if v <= 1.0:
        return v * 100.0
    return v

def predict_from_bytes(image_bytes: bytes, model_path: str = None):
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.exception("Failed to load ML model")
        raise RuntimeError(f"Failed to load ML model: {e}")

    tensor = _prepare_image(image_bytes).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)             # [1, num_classes]
        probs = F.softmax(logits, dim=1)   # [1, num_classes]
        probs_np = probs.squeeze(0).cpu().numpy().astype(float).tolist()

    # argmax index and prob
    top_idx = int(max(range(len(probs_np)), key=lambda i: probs_np[i]))
    top_prob = float(probs_np[top_idx])

    # safe class name mapping (CLASS_NAMES should reflect checkpoint's order if present)
    class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"class_{top_idx}"
    pretty = CLASS_DISPLAY_NAMES.get(class_name, class_name)

    # percentage: normalize to 0..100 (try helper first, but guarantee scale)
    percent = None
    try:
        percent_try = probs_to_percentage(probs_np)  # unknown scale from helper
        if percent_try is not None:
            percent_try = _normalize_percent_value(percent_try)
        if percent_try is not None:
            percent = round(percent_try, 2)
    except Exception:
        logger.debug("probs_to_percentage failed", exc_info=True)

    if percent is None:
        percent = round(top_prob * 100.0, 2)

    # severity derived in two ways:
    # 1) severity_by_bucket: using percentage_to_bucket (informational)
    severity_by_bucket = None
    try:
        # assume percentage_to_bucket expects 0..100; if it expects 0..1 pass normalized
        severity_by_bucket = percentage_to_bucket(percent)
    except Exception:
        try:
            severity_by_bucket = percentage_to_bucket(percent / 100.0)
        except Exception:
            severity_by_bucket = None

    # 2) severity_label: deterministic mapping from predicted class (safe for UI)
    severity_label = pretty  # ex: 'BAIXO'/'MÉDIO'/'ALTO' from CLASS_DISPLAY_NAMES

    # gather checkpoint mapping if present for debugging
    ckpt_info = None
    try:
        ckpt = torch.load(model_path or MODEL_PATH, map_location='cpu')
        if isinstance(ckpt, dict) and ("class_to_idx" in ckpt or "class_mapping" in ckpt):
            ckpt_info = ckpt.get("class_to_idx") or ckpt.get("class_mapping")
    except Exception:
        ckpt_info = getattr(model, "_ckpt_class_to_idx", None)

    response = {
        "class_index": top_idx,
        "class_name": class_name,
        "class_pretty": pretty,
        "prob": round(top_prob, 4),
        "probs": [round(float(p), 4) for p in probs_np],
        "percentage": percent,                       # ALWAYS 0..100
        "severity_label": severity_label,            # deterministic-friendly label
        "severity_by_bucket": severity_by_bucket,    # optional extra info (from percentage_to_bucket)
        "classes": list(CLASS_NAMES),                # mapping index -> class_name (helps frontend)
        "model_version": os.path.basename(model_path or MODEL_PATH),
        "ckpt_class_to_idx": ckpt_info
    }

    # sanity logs (optional)
    logger.debug("predict_from_bytes: top_idx=%s class_name=%s top_prob=%s percent=%s", top_idx, class_name, top_prob, percent)
    logger.debug("predict_from_bytes: probs=%s", [round(float(p), 6) for p in probs_np])
    logger.debug("predict_from_bytes: response (to be returned) = %s", response)
    logger.debug("Prediction response: idx=%s name=%s prob=%s percent=%s", top_idx, class_name, top_prob, percent)
    return response
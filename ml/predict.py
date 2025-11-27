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
    global _model, NUM_CLASSES
    if _model is not None:
        return _model
    p = path or MODEL_PATH
    device = DEVICE
    model = _build_model(NUM_CLASSES)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model file not found: {p}")
    state = torch.load(p, map_location=device)
    # detect possible containers
    if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
        sd = state.get("state_dict") or state.get("model_state_dict")
        # optionally load class_to_idx saved in checkpoint
        class_to_idx = state.get("class_to_idx") or state.get("class_mapping")
        if class_to_idx:
            # se existir, reindex CLASS_NAMES para refletir ordem usada no treino
            try:
                # class_to_idx: {class_name: idx}
                inv = {v: k for k, v in class_to_idx.items()}
                # build CLASS_NAMES_ORDERED local copy
                global CLASS_NAMES
                CLASS_NAMES = [inv[i] for i in range(len(inv))]
                logger.debug("Loaded class_to_idx from checkpoint; CLASS_NAMES adjusted.")
            except Exception as e:
                logger.debug("Failed to apply class_to_idx from checkpoint: %s", e)
        model.load_state_dict(sd)
    else:
        # state is raw state_dict or other
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model = model
    return _model

def _prepare_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0)  # [1,C,H,W]

def predict_from_bytes(image_bytes: bytes, model_path: str = None):
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.exception("Failed to load ML model")
        raise RuntimeError(f"Failed to load ML model: {e}")

    tensor = _prepare_image(image_bytes).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)              # [1, num_classes]
        probs = F.softmax(logits, dim=1)    # [1, num_classes]
        probs_np = probs.squeeze(0).cpu().numpy().tolist()

    # top class
    top_idx = int(max(range(len(probs_np)), key=lambda i: probs_np[i]))
    top_prob = float(probs_np[top_idx])
    # safe mapping: if CLASS_NAMES length mismatch, fallback to index-based name
    class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"class_{top_idx}"
    pretty = CLASS_DISPLAY_NAMES.get(class_name, class_name)

    # percentage using provided helper, fallback to top_prob*100
    try:
        percent = probs_to_percentage(probs_np)
    except Exception:
        logger.debug("probs_to_percentage failed; using top_prob*100 fallback")
        percent = round(top_prob * 100, 2)

    severity = percentage_to_bucket(percent)

    # return also class_to_idx info if available (helps debug)
    ckpt_info = None
    try:
        ckpt = torch.load(model_path or MODEL_PATH, map_location='cpu')
        if isinstance(ckpt, dict) and ("class_to_idx" in ckpt or "class_mapping" in ckpt):
            ckpt_info = ckpt.get("class_to_idx") or ckpt.get("class_mapping")
    except Exception:
        ckpt_info = None

    return {
        "class_index": top_idx,
        "class_name": class_name,
        "class_pretty": pretty,
        "prob": round(top_prob, 4),
        "probs": [round(float(p), 4) for p in probs_np],
        "percentage": percent,
        "severity_label": severity,
        "model_version": os.path.basename(model_path or MODEL_PATH),
        "ckpt_class_to_idx": ckpt_info
    }

"""
eval.py

Avalia o modelo treinado sobre a pasta ml/data/test (ImageFolder).
Gera relatório de classificação e matriz de confusão (opcional: salva figura %).
"""

import os
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from utils import CLASS_NAMES

# ----------------------
# Helpers
# ----------------------
def _get_num_workers():
    # Em Windows, usar 0 evita problemas; em Linux/Mac podemos usar >0
    return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

def load_model_checkpoint(path, num_classes, device, build_fn=None):
    """
    Carrega um modelo compatível com o usado no treino.
    - build_fn: função que retorna a arquitetura (ex.: train.build_model / model.build)
      deve aceitar (num_classes=..., pretrained=False) ou similar.
    """
    if build_fn is None:
        try:
            from train import build_model as _build_model
        except Exception:
            try:
                from ml.train import build_model as _build_model
            except Exception:
                _build_model = None
        build_fn = _build_model

    if build_fn is None:
        raise RuntimeError("Nenhuma função de construção de modelo (build_model) encontrada. "
                           "Passe build_fn ou verifique ml/train.py")

    model = build_fn(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(path, map_location=device)

    # checkpoint pode ser um state_dict direto ou um dict com 'state_dict'/'model_state_dict'
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        model.load_state_dict(sd)
    else:
        # assume que é um state_dict padrão
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model

def plot_confusion_matrix_pct(cm, class_names, out_path):
    """
    Recebe confusion_matrix (não normalizada) e plota a versão em porcentagem por linha (recall %).
    Salva PNG em out_path.
    """
    # Normaliza por linha (true labels) -> recall %
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = (cm.astype(float) / (row_sums + 1e-12)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion matrix (percent by true class)'
    )

    # Rotaciona as labels no eixo x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anota percentual em cada célula
    fmt = ".1f"
    thresh = cm_pct.max() / 2.0 if cm_pct.size else 0.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            text = f"{cm_pct[i, j]:.1f}%\n({int(cm[i,j])})"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm_pct[i, j] > thresh else "black", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ----------------------
# Main
# ----------------------
def main(model_path="ml/models/best.pth",
         data_dir="ml/data",
         batch_size=32,
         device_str="cpu",
         plot_path=None,
         verbose=1):
    device = torch.device(device_str)
    num_workers = _get_num_workers()

    # Transforms devem ser os mesmos usados no treino/val
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    test_folder = os.path.join(data_dir, "test")
    if not os.path.isdir(test_folder):
        raise FileNotFoundError(f"Test folder not found: {test_folder}")

    test_ds = datasets.ImageFolder(test_folder, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device_str.startswith("cuda")))

    if verbose >= 1:
        print("Class names:", CLASS_NAMES)
        print("ImageFolder class_to_idx:", test_ds.class_to_idx)
        print(f"Using device: {device}")

    # Tenta carregar o modelo usando função de build do train.py
    try:
        model = load_model_checkpoint(model_path, num_classes=len(CLASS_NAMES), device=device)
    except Exception as e:
        # fallback: tenta função build_model definida em ml/train.py (se não capturada antes),
        # ou tenta construir manually com torchvision (resnet18) caso não exista build_model.
        if verbose >= 1:
            print("load_model_checkpoint falhou:", e)
            print("Tentando alternativa de build automática (resnet18)...")
        from torchvision import models
        model = models.resnet18(weights=None)
        num_f = model.fc.in_features
        model.fc = torch.nn.Linear(num_f, len(CLASS_NAMES))
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
            sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
            model.load_state_dict(sd)
        else:
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds, dtype=int)
    all_targets = np.array(all_targets, dtype=int)

    cm = confusion_matrix(all_targets, all_preds)
    if verbose >= 1:
        print("Confusion matrix (counts):")
        print(cm)
        print("Classification report:")
        print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))

    if plot_path:
        try:
            plot_confusion_matrix_pct(cm, CLASS_NAMES, plot_path)
            if verbose >= 1:
                print("Saved confusion matrix plot to:", plot_path)
        except Exception as e:
            print("Falha ao gerar plot da matriz de confusão:", e)

    # verbose 2: detalhes extras
    if verbose >= 2:
        unique, counts = np.unique(all_targets, return_counts=True)
        print("Test set class distribution (true labels):")
        for u, c in zip(unique, counts):
            print(f"  {CLASS_NAMES[u]}: {c} samples")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate trained model on ml/data/test")
    p.add_argument("--model-path", default="ml/models/best.pth")
    p.add_argument("--data-dir", default="ml/data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--plot", default=None, help="save confusion matrix plot (PNG path)")
    p.add_argument("--verbose", type=int, default=1, choices=[0,1,2], help="0=min,1=normal,2=debug")
    args = p.parse_args()

    main(model_path=args.model_path,
         data_dir=args.data_dir,
         batch_size=args.batch_size,
         device_str=args.device,
         plot_path=args.plot,
         verbose=args.verbose)

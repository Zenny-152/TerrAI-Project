import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from utils import CLASS_NAMES

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224):
    transform_train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    transform_val = T.Compose([
        T.Resize(int(img_size*1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    return transform_train, transform_val

def build_model(num_classes, pretrained=True, base='resnet18'):
    if base == 'resnet18':
        # torchvision >= 0.13 may use weights=... instead of pretrained bool
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            model = models.resnet18(pretrained=pretrained)
        num_f = model.fc.in_features
        model.fc = nn.Linear(num_f, num_classes)
        return model
    # add other backbones if desired
    raise NotImplementedError

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print("Using device:", device)

    transform_train, transform_val = get_transforms(args.img_size)
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(args.data, "val"), transform_val)

    print("ImageFolder class_to_idx:", train_ds.class_to_idx)
    # verify order corresponds to CLASS_NAMES or adapt accordingly

    # compute class weights to mitigate imbalance
    targets = [y for _, y in train_ds.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained)
    # freeze backbone if requested
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    scaler = GradScaler() if args.amp else None

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best.pth")
    latest_path = os.path.join(args.save_dir, "latest.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                loss_v = criterion(out, labels)
                val_loss += loss_v.item()
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        avg_train_loss = running_loss / len(train_loader) if len(train_loader)>0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader)>0 else 0.0
        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")

        scheduler.step(avg_val_loss)
        # save
        torch.save({'state_dict': model.state_dict(), 'class_to_idx': train_ds.class_to_idx}, latest_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'class_to_idx': train_ds.class_to_idx}, best_path)
            print("Saved best model ->", best_path)

    print("Training finished. Best val acc:", best_acc)
    print("Latest saved at:", latest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml/data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="ml/models")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action='store_true')
    parser.add_argument("--no-pretrained", action='store_true')
    parser.add_argument("--amp", action='store_true', help="use mixed precision")
    args = parser.parse_args()
    main(args)

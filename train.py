"""
Train an EfficientNetB2 snake classifier using two-phase transfer learning.

Phase 1: Freeze the backbone, train the classification head.
Phase 2: Unfreeze the last two backbone stages and fine-tune at a lower LR.

Usage:
    python train.py
    python train.py --epochs1 10 --epochs2 15 --batch 16
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import timm

from config import (
    BATCH_SIZE, CLASSES, CONFIG_PATH, DATA_DIR, DEVICE,
    DROPOUT, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    LABEL_SMOOTHING, MODEL_NAME, MODEL_PATH, MODELS_DIR,
    PATIENCE, PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    TRAIN_RATIO, VAL_RATIO, WEIGHT_DECAY,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Transforms ─────────────────────────────────────────────────────────────────

_train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.06),
    transforms.RandomGrayscale(p=0.03),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

_val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Dataset ────────────────────────────────────────────────────────────────────

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_samples(root: Path) -> list[tuple[Path, int]]:
    """Recursively collect (image_path, class_idx) tuples from root/venomous and root/non_venomous."""
    samples = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found — run download_data.py first")
            continue
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in EXTS:
                samples.append((p, class_idx))
    return samples


class SnakeDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(data_dir: Path, batch_size: int):
    samples = _collect_samples(data_dir)
    if not samples:
        raise RuntimeError(
            "No images found in data/raw/venomous or data/raw/non_venomous.\n"
            "Run:  python download_data.py"
        )

    labels = [s[1] for s in samples]
    class_counts = [labels.count(0), labels.count(1)]
    print(f"  Dataset: {class_counts[0]:,} non-venomous  |  {class_counts[1]:,} venomous")

    # Stratified split
    indices = list(range(len(samples)))
    idx_train, idx_tmp = train_test_split(indices, test_size=1 - TRAIN_RATIO,
                                          random_state=SEED, stratify=labels)
    tmp_labels = [labels[i] for i in idx_tmp]
    val_frac = VAL_RATIO / (1 - TRAIN_RATIO)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=1 - val_frac,
                                         random_state=SEED, stratify=tmp_labels)

    train_samples = [samples[i] for i in idx_train]
    val_samples   = [samples[i] for i in idx_val]
    test_samples  = [samples[i] for i in idx_test]

    print(f"  Split  : {len(train_samples):,} train | {len(val_samples):,} val | {len(test_samples):,} test")

    # Balanced sampler for training
    train_labels = [s[1] for s in train_samples]
    cls_weights = 1.0 / torch.tensor(
        [train_labels.count(i) for i in range(len(CLASSES))], dtype=torch.float
    )
    sample_weights = cls_weights[[s[1] for s in train_samples]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    num_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(SnakeDataset(train_samples, _train_tf),
                              batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(SnakeDataset(val_samples, _val_tf),
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(SnakeDataset(test_samples, _val_tf),
                              batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class SnakeClassifier(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=0, global_pool="avg")
        n_feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Dropout(dropout),
            nn.Linear(n_feat, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_stages(self, n: int = 2):
        """Unfreeze the last n stages of the EfficientNet backbone."""
        stages = list(self.backbone.blocks)
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
        # Always unfreeze the final conv + norm layers
        for module in [self.backbone.conv_head, self.backbone.bn2]:
            for p in module.parameters():
                p.requires_grad = True


# ── Training helpers ───────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = correct = total = 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc="train" if train else "val  ", ncols=80, leave=False):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu()
            preds = (probs > 0.5).float()
            lbl_cpu = labels.detach().cpu()

            total_loss += loss.item() * len(labels)
            correct    += (preds == lbl_cpu).sum().item()
            total      += len(labels)
            all_probs.extend(probs.numpy())
            all_labels.extend(lbl_cpu.numpy())

    avg_loss = total_loss / total
    acc      = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, acc, auc


def train_phase(model, train_loader, val_loader, optimizer, scheduler,
                criterion, epochs, device, patience, phase_name):
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc, vl_auc = run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "vl_loss": vl_loss, "vl_acc": vl_acc, "vl_auc": vl_auc})

        print(f"  [{phase_name}] Ep {epoch:02d}/{epochs}  "
              f"tr {tr_loss:.4f}/{tr_acc:.3f}  "
              f"val {vl_loss:.4f}/{vl_acc:.3f}  AUC {vl_auc:.3f}  "
              f"({elapsed:.0f}s)")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", type=int, default=PHASE1_EPOCHS)
    parser.add_argument("--epochs2", type=int, default=PHASE2_EPOCHS)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE)
    parser.add_argument("--model",   type=str, default=MODEL_NAME)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Snake Classifier  |  device: {DEVICE}  |  model: {args.model}")
    print(f"{'='*60}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    train_loader, val_loader, test_loader = make_loaders(DATA_DIR, args.batch)

    model = SnakeClassifier(model_name=args.model).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None, reduction="mean")

    # ── Phase 1: train head only ─────────────────────────────────────────────
    print(f"\n── Phase 1: head only ({args.epochs1} epochs max) ──")
    model.freeze_backbone()
    optimizer1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    sched1 = CosineAnnealingLR(optimizer1, T_max=args.epochs1, eta_min=PHASE1_LR * 0.01)
    hist1 = train_phase(model, train_loader, val_loader, optimizer1, sched1,
                        criterion, args.epochs1, DEVICE, PATIENCE, "P1")

    # ── Phase 2: fine-tune last stages ──────────────────────────────────────
    print(f"\n── Phase 2: fine-tune last 2 backbone stages ({args.epochs2} epochs max) ──")
    model.unfreeze_last_stages(n=2)
    optimizer2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    sched2 = CosineAnnealingLR(optimizer2, T_max=args.epochs2, eta_min=PHASE2_LR * 0.01)
    hist2 = train_phase(model, train_loader, val_loader, optimizer2, sched2,
                        criterion, args.epochs2, DEVICE, PATIENCE, "P2")

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n── Test set evaluation ──")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, ncols=80):
            imgs   = imgs.to(DEVICE)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    print(f"\n  Test accuracy : {acc * 100:.2f}%")
    print(f"  Test AUC      : {auc:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # ── Save model ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "model_name": args.model,
        "image_size": IMAGE_SIZE,
        "classes": CLASSES,
        "test_accuracy": round(acc * 100, 2),
        "test_auc": round(auc, 4),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    CONFIG_PATH.write_text(json.dumps(meta, indent=2))

    print(f"\nModel saved to  : {MODEL_PATH}")
    print(f"Config saved to : {CONFIG_PATH}")
    print("\nNext step:  python app.py")


if __name__ == "__main__":
    main()

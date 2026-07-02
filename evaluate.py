"""
Evaluate the trained model and generate visualisation plots.

Usage:
    python evaluate.py
    python evaluate.py --samples 16    # show 16 sample predictions
"""
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm

from config import (
    CLASSES, CONFIG_PATH, DATA_DIR, DEVICE, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH, MODELS_DIR, PLOTS_DIR,
)
from train import SnakeClassifier, SnakeDataset, _collect_samples, _val_tf


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}.\nRun:  python train.py"
        )
    meta = json.loads(CONFIG_PATH.read_text())
    model = SnakeClassifier(model_name=meta["model_name"]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, meta = load_model()
    print(f"Model: {meta['model_name']}  |  trained accuracy: {meta.get('test_accuracy', '?')}%")

    # Build test set (last 10% of all samples, deterministic seed)
    from sklearn.model_selection import train_test_split
    samples = _collect_samples(DATA_DIR)
    labels  = [s[1] for s in samples]
    _, idx_tmp = train_test_split(range(len(samples)), test_size=0.20, random_state=42, stratify=labels)
    tmp_labels = [labels[i] for i in idx_tmp]
    _, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=tmp_labels)
    test_samples = [samples[i] for i in idx_test]

    loader = DataLoader(SnakeDataset(test_samples, _val_tf),
                        batch_size=64, shuffle=False, num_workers=2)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Evaluating", ncols=80):
            imgs   = imgs.to(DEVICE)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nTest accuracy : {acc*100:.2f}%")
    print(f"Test AUC      : {auc:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1 — Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # 2 — ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # 3 — Confidence distribution
    v_probs  = [p for p, l in zip(all_probs, all_labels) if l == 1]
    nv_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
    axes[2].hist(v_probs,  bins=40, alpha=0.6, label="Venomous",     color="#ef4444")
    axes[2].hist(nv_probs, bins=40, alpha=0.6, label="Non-venomous", color="#22c55e")
    axes[2].axvline(0.5, color="k", linestyle="--", lw=1)
    axes[2].set_xlabel("Predicted probability (venomous)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Confidence Distribution")
    axes[2].legend()

    plt.tight_layout()
    out = PLOTS_DIR / "evaluation.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlots saved to {out}")

    # ── Sample predictions grid ───────────────────────────────────────────────
    n = min(args.samples, len(test_samples))
    chosen = random.sample(range(len(test_samples)), n)
    cols = 4
    rows = math.ceil(n / cols)
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes2 = axes2.flatten()

    with torch.no_grad():
        for ax_i, sample_i in enumerate(chosen):
            path, true_label = test_samples[sample_i]
            img = Image.open(path).convert("RGB")
            tensor = _val_tf(img).unsqueeze(0).to(DEVICE)
            prob = torch.sigmoid(model(tensor).squeeze()).item()
            pred = int(prob > 0.5)
            color = "#22c55e" if pred == true_label else "#ef4444"
            label_str = f"{'Venomous' if pred else 'Safe'}  {prob*100:.0f}%"
            axes2[ax_i].imshow(img)
            axes2[ax_i].set_title(label_str, color=color, fontsize=9, pad=2)
            axes2[ax_i].axis("off")

    for ax in axes2[n:]:
        ax.axis("off")

    plt.tight_layout()
    out2 = PLOTS_DIR / "sample_predictions.png"
    plt.savefig(out2, dpi=150)
    print(f"Sample predictions saved to {out2}")


if __name__ == "__main__":
    main()

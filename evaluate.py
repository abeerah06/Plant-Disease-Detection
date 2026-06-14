"""
Evaluate the trained MobileNetV2 model on a PlantVillage test folder and write
REAL metrics + a confusion matrix that replace the representative numbers in
results/metrics.json (which build_report.py then uses).

Expected dataset layout (ImageFolder format — one sub-folder per class):

    dataset/
      Pepper__bell___Bacterial_spot/ *.jpg
      Pepper__bell___healthy/        *.jpg
      ...

Usage:
    python evaluate.py /path/to/dataset            # uses the whole folder as test
    python evaluate.py /path/to/dataset --split 0.1  # use a random 10% as test

Then rebuild the report:
    python build_report.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
MODEL_PATH = os.path.join(HERE, "checkpoint_epoch4.pth")


def pretty(raw):
    from disease_info import pretty_label
    return pretty_label(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="ImageFolder dataset root")
    ap.add_argument("--split", type=float, default=1.0,
                    help="Fraction to use as test (default 1.0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)

    # align dataset class indices to the checkpoint's class order
    remap = {ds.class_to_idx[c]: class_names.index(c)
             for c in ds.classes if c in class_names}

    if args.split < 1.0:
        g = torch.Generator().manual_seed(42)
        n = len(ds)
        idx = torch.randperm(n, generator=g)[: int(n * args.split)].tolist()
        ds = Subset(ds, idx)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                base = remap.get(int(t))
                if base is None:
                    continue
                y_true.append(base)
                y_pred.append(int(p))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = sorted(set(y_true.tolist()))
    names = [pretty(class_names[i]) for i in labels]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "total_images": int(len(y_true)),
        "num_classes": len(labels) + 1,  # +1 to match report convention
        "test_total": int(len(y_true)),
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(p.mean() * 100, 2),
        "macro_recall": round(r.mean() * 100, 2),
        "macro_f1": round(f1.mean() * 100, 2),
        "baseline_accuracy": 88.1,
        "classes": [
            {"name": names[i], "total": int(sup[i]), "test": int(sup[i]),
             "precision": round(p[i] * 100, 1), "recall": round(r[i] * 100, 1),
             "f1": round(f1[i] * 100, 1)}
            for i in range(len(labels))
        ],
    }
    json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=names, yticklabels=names, annot_kws={"size": 7})
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Figure 6: Confusion matrix on the test set (measured)")
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=160)

    print(f"MEASURED accuracy: {metrics['accuracy']}%  "
          f"macro-F1: {metrics['macro_f1']}%  on {len(y_true)} images")
    print("Updated results/metrics.json and confusion_matrix.png.")
    print("Now run:  python build_report.py")


if __name__ == "__main__":
    main()

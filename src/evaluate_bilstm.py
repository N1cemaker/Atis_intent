# src/evaluate_bilstm.py
import os
from typing import Set

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .models.bilstm import BiLSTMIntent


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    ckpt_path = "checkpoints/bilstm_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Please run `python -m src.train_bilstm` first."
        )


    ckpt = torch.load(ckpt_path, map_location=device)

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    cfg_dict = ckpt["config"]
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    dropout = ckpt["dropout"]

    max_len = cfg_dict.get("max_len", cfg.max_len)
    emb_dim = cfg_dict.get("emb_dim", cfg.emb_dim)


    test_sents, test_labels = load_processed_file(cfg.test_path)


    filtered_sents = []
    filtered_labels = []
    skipped = 0
    unseen_intents: Set[str] = set()

    for s, lab in zip(test_sents, test_labels):
        if lab in label2id:
            filtered_sents.append(s)
            filtered_labels.append(lab)
        else:
            skipped += 1
            unseen_intents.add(lab)

    if skipped > 0:
        print(
            f"[WARN] Skipped {skipped} test examples with intents unseen in training "
            f"(e.g., {unseen_intents})."
        )

    print(f"[Test] Using {len(filtered_sents)} examples after filtering.")

    test_ds = ATISDataset(filtered_sents, filtered_labels, stoi, label2id, max_len)
    test_loader = DataLoader(test_ds, batch_size=64)


    num_classes = len(label2id)
    vocab_size = len(stoi)

    model = BiLSTMIntent(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=stoi["<pad>"],
        num_layers=num_layers,
        bidirectional=True,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()


    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())


    all_label_ids = sorted(label2id.values())  
    target_names = [id2label[i] for i in all_label_ids]

    print("=== Classification Report (BiLSTM on ATIS test set) ===")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=all_label_ids,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )


    cm = confusion_matrix(all_labels, all_preds, labels=all_label_ids)
    print("=== Confusion Matrix Shape (counts):", cm.shape, " ===")

    os.makedirs("reports/figures", exist_ok=True)

 
    plt.figure(figsize=(10, 8))
    if USE_SEABORN:
        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            xticklabels=target_names,
            yticklabels=target_names,
        )
    else:
        plt.imshow(cm, interpolation="nearest", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(target_names)), target_names, rotation=90)
        plt.yticks(range(len(target_names)), target_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("BiLSTM Confusion Matrix on ATIS (test) - Counts")
    plt.tight_layout()
    fig_counts = "reports/figures/bilstm_confusion_matrix_counts.png"
    plt.savefig(fig_counts, dpi=300)
    plt.close()
    print(f"[Figure] Count confusion matrix saved to {fig_counts}")

    
    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    print("=== Confusion Matrix Shape (normalized):", cm_norm.shape, " ===")

    plt.figure(figsize=(10, 8))
    if USE_SEABORN:
        sns.heatmap(
            cm_norm,
            annot=False,
            fmt=".2f",
            xticklabels=target_names,
            yticklabels=target_names,
            vmin=0.0,
            vmax=1.0,
        )
    else:
        plt.imshow(cm_norm, interpolation="nearest", aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xticks(range(len(target_names)), target_names, rotation=90)
        plt.yticks(range(len(target_names)), target_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("BiLSTM Normalized Confusion Matrix on ATIS (test)")
    plt.tight_layout()
    fig_norm = "reports/figures/bilstm_confusion_matrix_norm.png"
    plt.savefig(fig_norm, dpi=300)
    plt.close()
    print(f"[Figure] Normalized confusion matrix saved to {fig_norm}")


if __name__ == "__main__":
    main()

# src/evaluate_textcnn.py
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
from .models.textcnn import TextCNN


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    ckpt_path = "/root/atis_intent/checkpoints/textcnn_classweight_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Please run `python -m src.train_textcnn` first."
        )

    # 0) load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    cfg_dict = ckpt["config"]
    filter_sizes = ckpt["filter_sizes"]
    num_filters = ckpt["num_filters"]
    dropout = ckpt["dropout"]

    max_len = cfg_dict.get("max_len", cfg.max_len)
    emb_dim = cfg_dict.get("emb_dim", cfg.emb_dim)

    # 1) load test data（processed/test.txt）
    test_sents, test_labels = load_processed_file(cfg.test_path)

    # 1.1 cut unknow intent
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

    # 2) Create TextCNN
    num_classes = len(label2id)
    vocab_size = len(stoi)

    model = TextCNN(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_classes=num_classes,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        pad_idx=stoi["<pad>"],
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) Predict on Test
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

    # 4) create label id list
    all_label_ids = sorted(label2id.values()) 
    target_names = [id2label[i] for i in all_label_ids]

    print("=== Classification Report (TextCNN on ATIS test set) ===")
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

    # 5) Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=all_label_ids)
    print("=== Confusion Matrix Shape (counts):", cm.shape, " ===")

    os.makedirs("reports/figures", exist_ok=True)

    # 5.1 Init Confusion Matrix
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
    plt.title("TextCNN Confusion Matrix on ATIS (test)")
    plt.tight_layout()
    fig_path_counts = "reports/figures/textcnn_weight_confusion_matrix_counts.png"
    plt.savefig(fig_path_counts, dpi=300)
    plt.close()
    print(f"[Figure] Count confusion matrix saved to {fig_path_counts}")

    # 5.2 L Confusion Matrix
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
    plt.title("TextCNN Normalized Confusion Matrix on ATIS (test)")
    plt.tight_layout()
    fig_path_norm = "reports/figures/textcnn_weight_confusion_matrix_norm.png"
    plt.savefig(fig_path_norm, dpi=300)
    plt.close()
    print(f"[Figure] Normalized confusion matrix saved to {fig_path_norm}")


if __name__ == "__main__":
    main()

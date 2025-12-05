# src/train_textcnn.py
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .vocab import build_vocab, build_label_mapping
from .models.textcnn import TextCNN
from .utils import evaluate


def build_class_weights(
    train_labels,
    label2id: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    y_train_ids = np.array([label2id[lbl] for lbl in train_labels])

    classes = np.array(sorted(label2id.values()))  # 0,1,...,K-1
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_ids,
    )

    print("[Class Weights]")
    for i, w in enumerate(class_weights_np):
        print(f"  id={i:2d}  intent={id2id[i]:25s}  weight={w:.4f}")

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    return class_weights


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # ---------------------------------------------------------
    # 1. load train / valid data
    # ---------------------------------------------------------
    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)

    # ---------------------------------------------------------
    # 2. create vocab & label mapping
    # ---------------------------------------------------------
    stoi, itos = build_vocab(train_sents, min_freq=cfg.min_freq)
    label2id, id2label = build_label_mapping(train_labels)

    # ---------------------------------------------------------
    # 3. create Dataset / DataLoader
    # ---------------------------------------------------------
    train_ds = ATISDataset(train_sents, train_labels, stoi, label2id, cfg.max_len)
    valid_ds = ATISDataset(valid_sents, valid_labels, stoi, label2id, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)

    # ---------------------------------------------------------
    # 4. calculate class weights
    # ---------------------------------------------------------
    y_train_ids = np.array([label2id[lbl] for lbl in train_labels])
    classes = np.array(sorted(label2id.values()))
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_ids,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    print("[Class Weights]")
    for i, w in enumerate(class_weights_np):
        print(f"  id={i:2d}  intent={id2label[i]:25s}  weight={w:.4f}")

    # ---------------------------------------------------------
    # 5. init TextCNN
    # ---------------------------------------------------------
    num_classes = len(label2id)
    vocab_size = len(stoi)

    filter_sizes = [3, 4, 5]
    num_filters = 100
    dropout = 0.5

    model = TextCNN(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        num_classes=num_classes,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        pad_idx=stoi["<pad>"],
        dropout=dropout,
    ).to(device)

    print(model)

    # ---------------------------------------------------------
    # 6. loss function
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=0.0, 
    )

    # ---------------------------------------------------------
    # 7. training
    # ---------------------------------------------------------
    best_val_acc = 0.0
    best_state: Dict[str, Any] = {}
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/textcnn_classweight_best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)           # [B, num_classes]
            loss = criterion(logits, y) # scalar
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_train_loss = total_loss / total_examples
        val_acc = evaluate(model, valid_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_acc={val_acc:.4f}")

        # save the best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "label2id": label2id,
                "id2label": id2label,
                "config": cfg.__dict__,
                "filter_sizes": filter_sizes,
                "num_filters": num_filters,
                "dropout": dropout,
            }
            torch.save(best_state, save_path)
            print(f"  -> New best model saved to {save_path} with val_acc={best_val_acc:.4f}")

    print(f"Training finished. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

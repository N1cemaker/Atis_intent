# src/train_textcnn_reg_sweep.py
"""
Regularization sweep for TextCNN on ATIS:
- Dropout sweep (with fixed weight_decay=0.0)
- Weight decay sweep (with fixed dropout=0.5)

For each setting:
    * Train TextCNN from scratch
    * Save best checkpoint
    * Evaluate on test set
    * Record Val Acc / Test Acc for later table

Usage (from project root):
    python -m src.train_textcnn_reg_sweep
"""

import os
import json
from typing import List, Dict, Any, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .vocab import build_vocab, build_label_mapping
from .models.textcnn import TextCNN

DROPOUT_LIST: List[float] = [0.0, 0.3, 0.5, 0.7]      # TextCNN dropout
WD_LIST: List[float] = [0.0, 1e-4, 1e-3]             # Adam weight_decay


def evaluate_epoch(model, data_loader, criterion, device):
    """Compute avg loss and accuracy on given loader (val/test)."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / total_examples
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate_train_acc(model, data_loader, device):
    """Compute train accuracy only (no loss)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return accuracy_score(all_labels, all_preds)


def build_test_loader(cfg: Config, stoi, label2id, max_len, device):
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
    return test_loader


def train_textcnn_with_reg(
    base_cfg: Config,
    device: torch.device,
    dropout: float,
    weight_decay: float,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"[REG SWEEP] Training TextCNN with dropout={dropout}, weight_decay={weight_decay}")
    print("=" * 70)

    cfg = Config()
    cfg.__dict__.update(base_cfg.__dict__)

    # 1) train / valid
    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)

    # 2) vocab & labels
    stoi, itos = build_vocab(train_sents, min_freq=cfg.min_freq)
    label2id, id2label = build_label_mapping(train_labels)

    # 3) DataLoader
    train_ds = ATISDataset(train_sents, train_labels, stoi, label2id, cfg.max_len)
    valid_ds = ATISDataset(valid_sents, valid_labels, stoi, label2id, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)

    # 4) Test loader
    test_loader = build_test_loader(cfg, stoi, label2id, cfg.max_len, device)

    # 5) model
    num_classes = len(label2id)
    vocab_size = len(stoi)
    filter_sizes = [3, 4, 5]
    num_filters = 100

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

    # 6) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=weight_decay,  
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_val_acc = 0.0
    best_state = None

    train_accs, val_accs = [], []

    # 7) Training
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

        avg_train_loss = total_loss / total_examples
        train_acc = evaluate_train_acc(model, train_loader, device)
        avg_val_loss, val_acc = evaluate_epoch(model, valid_loader, criterion, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"[dropout={dropout}, wd={weight_decay}] Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}"
        )

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

    # 8) best state
    if best_state is None:
        raise RuntimeError("No best_state saved, something is wrong.")

    model.load_state_dict(best_state["model_state"])
    _, test_acc = evaluate_epoch(model, test_loader, criterion, device)

    print(
        f"[RESULT] dropout={dropout}, weight_decay={weight_decay} | "
        f"best_val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f}"
    )

    # 9) log
    log_entry = {
        "dropout": dropout,
        "weight_decay": weight_decay,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_acc_curve": train_accs,
        "val_acc_curve": val_accs,
    }

    log_name = f"logs/textcnn_reg_dropout_{dropout}_wd_{weight_decay}.json"
    with open(log_name, "w") as f:
        json.dump(log_entry, f, indent=2)
    print(f"[LOG] Saved per-run log to {log_name}")

    return log_entry


def main():
    base_cfg = Config()
    device = torch.device(base_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    print(f"[REG SWEEP] Dropout list = {DROPOUT_LIST}")
    print(f"[REG SWEEP] Weight decay list = {WD_LIST}")
    print(f"[REG SWEEP] Base lr = {base_cfg.lr}")

    os.makedirs("logs", exist_ok=True)

    results: Dict[str, Any] = {
        "dropout_sweep": [],
        "weight_decay_sweep": [],
    }

    fixed_wd = 0.0
    print("\n" + "#" * 70)
    print(f"[SWEEP] Dropout sweep with fixed weight_decay={fixed_wd}")
    print("#" * 70)

    for dp in DROPOUT_LIST:
        entry = train_textcnn_with_reg(
            base_cfg=base_cfg,
            device=device,
            dropout=dp,
            weight_decay=fixed_wd,
        )
        results["dropout_sweep"].append(entry)

    fixed_dp = 0.5
    print("\n" + "#" * 70)
    print(f"[SWEEP] Weight decay sweep with fixed dropout={fixed_dp}")
    print("#" * 70)

    for wd in WD_LIST:
        entry = train_textcnn_with_reg(
            base_cfg=base_cfg,
            device=device,
            dropout=fixed_dp,
            weight_decay=wd,
        )
        results["weight_decay_sweep"].append(entry)

    summary_path = "logs/textcnn_reg_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUMMARY] All regularization sweep results saved to {summary_path}")


if __name__ == "__main__":
    main()

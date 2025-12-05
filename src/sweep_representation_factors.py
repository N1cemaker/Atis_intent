# src/sweep_representation_factors.py

import os
import json
from typing import Dict, Any, List, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .vocab import build_vocab, build_label_mapping
from .models.textcnn import TextCNN

MIN_FREQ_LIST = [1, 2, 3, 5]
EMB_DIM_LIST = [50, 100, 200]
MAX_LEN_LIST = [30, 50, 70]


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_examples = 0.0, 0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_examples += bs
            p = logits.argmax(dim=-1)
            preds += p.cpu().tolist()
            labels += y.cpu().tolist()
    return total_loss / total_examples, accuracy_score(labels, preds)


def evaluate_train_acc(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(dim=-1)
            preds += p.cpu().tolist()
            labels += y.cpu().tolist()
    return accuracy_score(labels, preds)


def build_test_loader(cfg, stoi, label2id, max_len):
    sents, labs = load_processed_file(cfg.test_path)
    fs, fl = [], []
    for s, lab in zip(sents, labs):
        if lab in label2id:
            fs.append(s)
            fl.append(lab)
    ds = ATISDataset(fs, fl, stoi, label2id, max_len)
    return DataLoader(ds, batch_size=64)


def train_textcnn_one_run(base_cfg, device, *, min_freq, emb_dim, max_len):
    cfg = Config()
    cfg.__dict__.update(base_cfg.__dict__)
    cfg.min_freq = min_freq
    cfg.emb_dim = emb_dim
    cfg.max_len = max_len

    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)

    stoi, itos = build_vocab(train_sents, min_freq=cfg.min_freq)
    vocab_size = len(stoi)

    label2id, id2label = build_label_mapping(train_labels)

    train_ds = ATISDataset(train_sents, train_labels, stoi, label2id, cfg.max_len)
    valid_ds = ATISDataset(valid_sents, valid_labels, stoi, label2id, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)
    test_loader = build_test_loader(cfg, stoi, label2id, cfg.max_len)

    num_classes = len(label2id)
    model = TextCNN(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        num_classes=num_classes,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        pad_idx=stoi["<pad>"],
        dropout=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    best_state = None
    train_accs, val_accs = [], []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss, total_examples = 0.0, 0
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

        train_acc = evaluate_train_acc(model, train_loader, device)
        _, val_acc = evaluate_epoch(model, valid_loader, criterion, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[min_freq={min_freq}, emb_dim={emb_dim}, max_len={max_len}] "
              f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    _, test_acc = evaluate_epoch(model, test_loader, criterion, device)

    return {
        "min_freq": min_freq,
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "max_len": max_len,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_acc_curve": train_accs,
        "val_acc_curve": val_accs,
    }


def main():
    base_cfg = Config()
    device = torch.device(base_cfg.device if torch.cuda.is_available() else "cpu")

    os.makedirs("logs", exist_ok=True)

    results: Dict[str, Any] = {
        "vocab_sweep": [],
        "emb_dim_sweep": [],
        "max_len_sweep": [],
    }

    for mf in MIN_FREQ_LIST:
        r = train_textcnn_one_run(base_cfg, device, min_freq=mf, emb_dim=base_cfg.emb_dim, max_len=base_cfg.max_len)
        results["vocab_sweep"].append(r)

    for ed in EMB_DIM_LIST:
        r = train_textcnn_one_run(base_cfg, device, min_freq=base_cfg.min_freq, emb_dim=ed, max_len=base_cfg.max_len)
        results["emb_dim_sweep"].append(r)

    for ml in MAX_LEN_LIST:
        r = train_textcnn_one_run(base_cfg, device, min_freq=base_cfg.min_freq, emb_dim=base_cfg.emb_dim, max_len=ml)
        results["max_len_sweep"].append(r)

    with open("logs/textcnn_representation_sweep_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved summary to logs/textcnn_representation_sweep_summary.json")


if __name__ == "__main__":
    main()

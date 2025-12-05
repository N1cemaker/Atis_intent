# src/train_bilstm.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .vocab import build_vocab, build_label_mapping
from .models.bilstm import BiLSTMIntent


def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / total_examples
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate_train_acc(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return accuracy_score(all_labels, all_preds)


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # 1) load train / valid data
    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)

    # 2) load vocab & label reflect
    stoi, itos = build_vocab(train_sents, min_freq=cfg.min_freq)
    label2id, id2label = build_label_mapping(train_labels)

    # 3) create DataLoader
    train_ds = ATISDataset(train_sents, train_labels, stoi, label2id, cfg.max_len)
    valid_ds = ATISDataset(valid_sents, valid_labels, stoi, label2id, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)

    # 4) init BiLSTM
    num_classes = len(label2id)
    vocab_size = len(stoi)

    hidden_dim = 128    
    num_layers = 1      
    dropout = 0.5

    model = BiLSTMIntent(
        vocab_size=vocab_size,
        emb_dim=cfg.emb_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=stoi["<pad>"],
        num_layers=num_layers,
        bidirectional=True,
        dropout=dropout,
    ).to(device)

    print(model)

    # 5) Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    best_val_acc = 0.0
    best_state = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ====== Training ======
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_train_loss = total_loss / total_examples
        train_acc = evaluate_train_acc(model, train_loader, device)
        avg_val_loss, val_acc = evaluate_epoch(model, valid_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}"
        )

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
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            }
            torch.save(best_state, "checkpoints/bilstm_best.pt")
            print(f"  -> New best BiLSTM model saved with val_acc={best_val_acc:.4f}")

    print(f"BiLSTM training finished. Best val_acc={best_val_acc:.4f}")

    # ====== Curve ======
    epochs = range(1, cfg.num_epochs + 1)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiLSTM Training/Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("reports/figures/bilstm_loss_curve.png", dpi=300)
    plt.close()

    # Train + Val Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("BiLSTM Train/Validation Accuracy")
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("reports/figures/bilstm_train_val_acc_curve.png", dpi=300)
    plt.close()

    print("[Figure] BiLSTM curves saved to reports/figures/bilstm_*.png")


if __name__ == "__main__":
    main()

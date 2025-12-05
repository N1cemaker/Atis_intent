# src/train_textcnn_glove.py

import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from .config import Config
from .dataset import load_processed_file, ATISDataset
from .vocab import build_vocab, build_label_mapping
from .models.textcnn import TextCNN
from .utils.load_glove import load_glove_embedding

MODE = "glove_frozen"     # "random" / "glove_frozen" / "glove_finetune"
GLOVE_PATH = "/root/atis_intent/data/glove.6B.100d.txt" 
# ======================================


def evaluate_epoch(model, data_loader, criterion, device):
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
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return accuracy_score(all_labels, all_preds)


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    print(f"[Embedding MODE] {MODE}")

    # 1) load train / valid data
    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)

    # 2) create vocab & label reflect
    stoi, itos = build_vocab(train_sents, min_freq=cfg.min_freq)
    label2id, id2label = build_label_mapping(train_labels)

    # 3) create DataLoader
    train_ds = ATISDataset(train_sents, train_labels, stoi, label2id, cfg.max_len)
    valid_ds = ATISDataset(valid_sents, valid_labels, stoi, label2id, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)

    # 4) init TextCNN model
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

    # 5) Processing embedding
    if MODE in ["glove_frozen", "glove_finetune"]:
        if not os.path.exists(GLOVE_PATH):
            raise FileNotFoundError(
                f"GloVe file not found at {GLOVE_PATH}. "
                f"Please download glove.6B.100d.txt and set GLOVE_PATH."
            )

        glove_matrix = load_glove_embedding(
            glove_path=GLOVE_PATH,
            stoi=stoi,
            emb_dim=cfg.emb_dim,
        )
        with torch.no_grad():
            model.embedding.weight.data.copy_(torch.tensor(glove_matrix))

        if MODE == "glove_frozen":
            model.embedding.weight.requires_grad = False
            print("[Embedding] Using GloVe (frozen).")
        else:
            model.embedding.weight.requires_grad = True
            print("[Embedding] Using GloVe (fine-tuned).")
    else:
        print("[Embedding] Using random initialized, trainable embeddings (baseline).")

    print(model)

    # 6) Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  
        lr=cfg.lr,
    )
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    mode_tag = MODE

    best_val_acc = 0.0
    best_state = None
    train_losses, val_losses = [], []
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

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"[{mode_tag}] Epoch {epoch:02d} | "
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
                "mode": MODE,
            }
            ckpt_path = f"checkpoints/textcnn_{mode_tag}_best.pt"
            torch.save(best_state, ckpt_path)
            print(f"  -> New best model saved to {ckpt_path} (val_acc={best_val_acc:.4f})")

    print(f"[{mode_tag}] Training finished. Best val_acc={best_val_acc:.4f}")

    # 8) Log
    log = {
        "mode": MODE,
        "val_best": best_val_acc,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }
    log_path = f"logs/textcnn_{mode_tag}_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{mode_tag}] Log saved to {log_path}")

    # 9) acc curve
    epochs = range(1, cfg.num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"TextCNN Train/Val Accuracy ({mode_tag})")
    plt.ylim(0.8, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig_path = f"reports/figures/textcnn_train_val_acc_{mode_tag}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[{mode_tag}] Acc curve saved to {fig_path}")


if __name__ == "__main__":
    main()

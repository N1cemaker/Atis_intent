# src/train_bert.py
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from transformers import BertTokenizer

from .config import Config
from .dataset import load_processed_file
from .vocab import build_label_mapping
from .models.bert_intent import BERTIntentClassifier


class ATISBERTDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: BertTokenizer,
        max_len: int,
        label2id: dict,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label_str = self.labels[idx]
        label_id = self.label2id[label_str]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(label_id, dtype=torch.long)


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    cfg = Config()
    device = torch.device("cpu") 
    print(f"[Device] Using {device}")

    train_sents, train_labels = load_processed_file(cfg.train_path)
    valid_sents, valid_labels = load_processed_file(cfg.valid_path)
    print(f"[Load] {cfg.train_path}: {len(train_sents)} examples")
    print(f"[Load] {cfg.valid_path}: {len(valid_sents)} examples")

    label2id, id2label = build_label_mapping(train_labels)
    num_classes = len(label2id)
    print(f"[Labels] Num intents (train) = {num_classes}")

    pretrained_name = "prajjwal1/bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    max_len = 64 

    train_ds = ATISBERTDataset(
        train_sents, train_labels, tokenizer, max_len, label2id
    )
    valid_ds = ATISBERTDataset(
        valid_sents, valid_labels, tokenizer, max_len, label2id
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False
    )

    model = BERTIntentClassifier(
        num_classes=num_classes,
        pretrained_model=pretrained_name,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    best_val = 0.0
    best_state_path = "checkpoints/bert_best.pt"

    train_accs = []
    val_accs = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        all_preds = []
        all_labels = []

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = evaluate_epoch(model, valid_loader, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_state_path)
            print(
                f"  -> New best BERT model saved to {best_state_path} (val_acc={best_val:.4f})"
            )

    print(f"BERT training finished. Best val_acc={best_val:.4f}")

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("BERT Train/Validation Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path = "reports/figures/bert_train_val_acc_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Figure] BERT train/val accuracy curve saved to {out_path}")


if __name__ == "__main__":
    main()

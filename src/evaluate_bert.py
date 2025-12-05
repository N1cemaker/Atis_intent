# src/evaluate_bert.py
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from transformers import BertTokenizer

from .dataset import load_processed_file
from .vocab import build_label_mapping
from .models.bert_intent import BERTIntentClassifier


class ATISBERTTestDataset(Dataset):

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


def main():
    device = torch.device("cpu")
    print(f"[Device] Using {device}")

    TRAIN_PATH = "data/processed/train.txt"
    TEST_PATH = "data/processed/test.txt"

    train_sents, train_labels = load_processed_file(TRAIN_PATH)
    print(f"[Load] {TRAIN_PATH}: {len(train_sents)} examples")
    test_sents, test_labels = load_processed_file(TEST_PATH)
    print(f"[Load] {TEST_PATH}: {len(test_sents)} examples")

    label2id, id2label = build_label_mapping(train_labels)
    num_classes = len(label2id)
    print(f"[Labels] Num intents (train) = {num_classes}")

    filtered_sents = []
    filtered_labels = []
    skipped = 0
    skipped_set = set()

    for sent, lab in zip(test_sents, test_labels):
        if lab not in label2id:
            skipped += 1
            skipped_set.add(lab)
            continue
        filtered_sents.append(sent)
        filtered_labels.append(lab)

    if skipped > 0:
        print(
            f"[WARN] Skipped {skipped} test examples with intents unseen in training "
            f"(e.g., {skipped_set})."
        )

    print(f"[Test] Using {len(filtered_sents)} examples after filtering.")

    pretrained_name = "prajjwal1/bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    max_len = 64

    test_ds = ATISBERTTestDataset(
        filtered_sents, filtered_labels, tokenizer, max_len, label2id
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = BERTIntentClassifier(
        num_classes=num_classes,
        pretrained_model=pretrained_name,
    ).to(device)

    ckpt_path = "checkpoints/bert_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Please run `python -m src.train_bert` first."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("=== Classification Report (BERT) ===")
    target_names = [id2label[i] for i in range(num_classes)]
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=target_names,
            zero_division=0,
        )
    )

    os.makedirs("reports/figures", exist_ok=True)

    labels = list(range(num_classes))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    # 8.1 counts
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("BERT Confusion Matrix (Counts)")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_counts_path = "reports/figures/bert_confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=300)
    plt.close()

    # 8.2 normalized
    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_norm,
        row_sums,
        out=np.zeros_like(cm_norm),
        where=row_sums != 0,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("BERT Confusion Matrix (Normalized)")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_norm_path = "reports/figures/bert_confusion_matrix_norm.png"
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()

    print(
        "[Figure] BERT confusion matrices saved to "
        f"{cm_counts_path} and {cm_norm_path}"
    )


if __name__ == "__main__":
    main()

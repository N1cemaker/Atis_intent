# src/dataset.py
from typing import List, Tuple
from torch.utils.data import Dataset
import torch

from .vocab import encode_sentence


def load_processed_file(path: str) -> Tuple[List[str], List[str]]:

    sentences: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:

                continue
            text, intent = parts
            sentences.append(text)
            labels.append(intent)

    print(f"[Load] {path}: {len(sentences)} examples")
    return sentences, labels


class ATISDataset(Dataset):

    def __init__(self,
                 sentences: List[str],
                 labels: List[str],
                 stoi,
                 label2id,
                 max_len: int = 50):
        assert len(sentences) == len(labels)
        self.sentences = sentences
        self.labels = labels
        self.stoi = stoi
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        text = self.sentences[idx]
        label_str = self.labels[idx]

        x = encode_sentence(text, self.stoi, self.max_len)    # [max_len]
        y = torch.tensor(self.label2id[label_str], dtype=torch.long)  # 标量

        return x, y

# src/vocab.py
from collections import Counter
from typing import List, Dict, Tuple
import torch

SPECIAL_TOKENS = ["<pad>", "<unk>"]


def tokenize(text: str) -> List[str]:
    return text.split()


def build_vocab(sentences: List[str], min_freq: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))

    stoi: Dict[str, int] = {}

    for tok in SPECIAL_TOKENS:
        stoi[tok] = len(stoi)

    for word, freq in counter.items():
        if freq >= min_freq:
            stoi[word] = len(stoi)

    itos: Dict[int, str] = {i: w for w, i in stoi.items()}
    print(f"[Vocab] Size = {len(stoi)} (including special tokens)")
    return stoi, itos


def build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted(set(labels))
    label2id = {lbl: i for i, lbl in enumerate(unique)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    print(f"[Labels] Num intents = {len(label2id)}")
    return label2id, id2label


def encode_sentence(text: str, stoi: Dict[str, int], max_len: int = 50) -> torch.Tensor:
    tokens = tokenize(text)
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens]

    ids = ids[:max_len]

    pad_id = stoi["<pad>"]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)

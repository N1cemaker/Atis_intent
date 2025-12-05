# src/utils/load_glove.py
import numpy as np


def load_glove_embedding(glove_path: str, stoi: dict, emb_dim: int = 100):
    vocab_size = len(stoi)
    embeddings = np.random.uniform(-0.05, 0.05, (vocab_size, emb_dim)).astype("float32")

    print(f"[GloVe] Loading from: {glove_path}")
    print(f"[GloVe] Vocab size = {vocab_size}, emb_dim = {emb_dim}")

    found = 0
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < emb_dim + 1:
                continue
            word = parts[0]
            vec = parts[1:]

            if word in stoi:
                idx = stoi[word]
                embeddings[idx] = np.asarray(vec, dtype="float32")
                found += 1

    print(f"[GloVe] Found pretrained vectors for {found} / {vocab_size} tokens.")
    return embeddings

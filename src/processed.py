# src/preprocess.py
import json
import os
import re
import random
from typing import List, Tuple

random.seed(42)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\d+", "<num>", text)
    return text


def load_rasa_json(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("rasa_nlu_data", {}).get("common_examples", [])
    sentences: List[str] = []
    intents: List[str] = []

    for ex in examples:
        text = ex.get("text", "").strip()
        intent = ex.get("intent", "").strip()
        if not text or not intent:
            continue
        text = normalize_text(text)
        sentences.append(text)
        intents.append(intent)

    return sentences, intents


def train_valid_split(
    sentences: List[str],
    intents: List[str],
    valid_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    assert len(sentences) == len(intents)
    n = len(sentences)
    indices = list(range(n))
    random.shuffle(indices)

    valid_size = int(n * valid_ratio)
    valid_idx = indices[:valid_size]
    train_idx = indices[valid_size:]

    train_sents = [sentences[i] for i in train_idx]
    train_labels = [intents[i] for i in train_idx]
    valid_sents = [sentences[i] for i in valid_idx]
    valid_labels = [intents[i] for i in valid_idx]

    return train_sents, train_labels, valid_sents, valid_labels


def save_processed(
    path: str,
    sentences: List[str],
    intents: List[str],
):
    assert len(sentences) == len(intents)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for text, intent in zip(sentences, intents):
            line = f"{text}\t{intent}\n"
            f.write(line)


def main():
    raw_train_path = os.path.join("data", "/root/atis_intent/data/train.json")
    raw_test_path  = os.path.join("data", "/root/atis_intent/data/test.json") 

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    train_sents_all, train_intents_all = load_rasa_json(raw_train_path)
    print(f"Loaded {len(train_sents_all)} training examples from {raw_train_path}")

    train_sents, train_labels, valid_sents, valid_labels = train_valid_split(
        train_sents_all, train_intents_all, valid_ratio=0.1
    )
    print(f"Train size: {len(train_sents)}, Valid size: {len(valid_sents)}")

    test_sents, test_labels = [], []
    if os.path.exists(raw_test_path):
        test_sents, test_labels = load_rasa_json(raw_test_path)
        print(f"Loaded {len(test_sents)} test examples from {raw_test_path}")
    else:
        print(f"[WARN] Test JSON not found at {raw_test_path}, using part of training as test.")
        train_sents, train_labels, test_sents, test_labels = train_valid_split(
            train_sents, train_labels, valid_ratio=0.1
        )
        print(f"New Train size: {len(train_sents)}, Test size: {len(test_sents)}")

    save_processed(os.path.join(processed_dir, "train.txt"), train_sents, train_labels)
    save_processed(os.path.join(processed_dir, "valid.txt"), valid_sents, valid_labels)
    save_processed(os.path.join(processed_dir, "test.txt"),  test_sents,  test_labels)

    print("Saved processed train/valid/test to data/processed/")


if __name__ == "__main__":
    main()

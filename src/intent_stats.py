import json
from collections import Counter
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    USE_SEABORN = True
except:
    USE_SEABORN = False


def count_intents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["rasa_nlu_data"]["common_examples"]
    counter = Counter(ex["intent"] for ex in examples)

    return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def plot_intent_bar(counter_dict, save_path="intent_counts_test.png"):
    intents = list(counter_dict.keys())
    counts = list(counter_dict.values())

    plt.figure(figsize=(12, 8))

    if USE_SEABORN:
        sns.barplot(x=counts, y=intents, orient="h")
    else:
        plt.barh(intents, counts)

    plt.xlabel("Count")
    plt.ylabel("Intent")
    plt.title("Intent Distribution in ATIS Dataset")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] Bar chart saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    json_path = "/root/atis_intent/data/test.json" 
    counter = count_intents(json_path)

    print("Intent counts:")
    for k, v in counter.items():
        print(f"{k:20s} {v}")

    plot_intent_bar(counter)

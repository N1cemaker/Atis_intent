import os
import json
import matplotlib.pyplot as plt

def plot_lr_curves(log_files, figsize=(10,6), save_path="reports/figures/lr_compare.png"):
    """
    log_files: dict, key = display name, value = json file path
    """
    plt.figure(figsize=figsize)

    for label, path in log_files.items():
        if not os.path.exists(path):
            print(f"[WARN] Log file not found: {path}")
            continue
        
        with open(path, "r") as f:
            log = json.load(f)

        val_acc = log["val_acc"]
        epochs = range(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc, marker="o", label=f"LR={log['lr']} ({label})")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("TextCNN Learning Rate Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.ylim(0.85, 1.0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[FIGURE] Saved to {save_path}")


def main():
    log_files = {
        "1e-4": "logs/textcnn_lr_0.0001.json",
        "5e-4": "logs/textcnn_lr_0.0005.json",
        "1e-3": "logs/textcnn_lr_0.001.json",
        "5e-3": "logs/textcnn_lr_0.005.json"
    }

    plot_lr_curves(log_files)


if __name__ == "__main__":
    main()

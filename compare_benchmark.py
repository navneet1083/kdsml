# compare_benchmarks.py
import os
import json
import matplotlib.pyplot as plt


def load_metrics(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f"File {filename} not found.")
        return None


def plot_comparison(glue_metrics, squad_metrics):
    # Assume each metrics dict contains train_losses, val_losses, and (for GLUE) val_accuracies.
    epochs_glue = range(1, len(glue_metrics["train_losses"]) + 1)
    epochs_squad = range(1, len(squad_metrics["train_losses"]) + 1)

    plt.figure(figsize=(14, 6))

    # Plot GLUE training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_glue, glue_metrics["train_losses"], label="GLUE Train Loss", marker="o")
    plt.plot(epochs_glue, glue_metrics["val_losses"], label="GLUE Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GLUE Loss")
    plt.legend()

    # Plot GLUE validation accuracy (if available)
    if "val_accuracies" in glue_metrics:
        plt.subplot(1, 3, 2)
        plt.plot(epochs_glue, glue_metrics["val_accuracies"], label="GLUE Val Acc", marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("GLUE Accuracy")
        plt.legend()

    # Plot SQuAD training and validation loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs_squad, squad_metrics["train_losses"], label="SQuAD Train Loss", marker="o")
    plt.plot(epochs_squad, squad_metrics["val_losses"], label="SQuAD Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SQuAD Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/benchmark_comparison.png")
    plt.show()
    print("Comparison plot saved as plots/benchmark_comparison.png")


def main():
    glue_metrics = load_metrics("glue_metrics.json")
    squad_metrics = load_metrics("squad_metrics.json")
    if glue_metrics is not None and squad_metrics is not None:
        plot_comparison(glue_metrics, squad_metrics)


if __name__ == "__main__":
    main()

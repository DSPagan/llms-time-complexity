"""
Evaluation utilities: turn raw model outputs into complexity classes and compute
accuracy, macro F1-score and the confusion matrix used throughout the thesis.

The model answers either in big-O notation (zero-/few-shot) or with the class
name directly (fine-tuned); ``normalize_prediction`` handles both. Predictions
that cannot be mapped to any class count as a failure for accuracy and F1 but are
excluded from the confusion matrix, following the thesis convention.
"""

import re

# Ordered as in the thesis confusion matrices.
CLASSES = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "exponential"]

# Big-O labels for plotting.
DISPLAY = {
    "constant": "O(1)",
    "logn": "O(logn)",
    "linear": "O(n)",
    "nlogn": "O(nlogn)",
    "quadratic": "O(n²)",
    "cubic": "O(n³)",
    "exponential": "exp",
}

UNCLASSIFIED = "unclassified"


def normalize_prediction(text):
    """Map a raw model answer to one of ``CLASSES`` (or ``UNCLASSIFIED``)."""
    if not text:
        return UNCLASSIFIED

    s = text.lower().replace("²", "^2").replace("³", "^3").replace("**", "^")
    s = s.replace(" ", "")

    # Order matters: check the most specific patterns first so that, e.g., "nlogn"
    # is not caught by the "logn" rule and "n^2" is not caught by the "n" rule.
    if "exponential" in s or re.search(r"[0-9k]\^n", s):
        return "exponential"
    if "cubic" in s or "n^3" in s:
        return "cubic"
    if "quadratic" in s or "n^2" in s:
        return "quadratic"
    if "nlogn" in s or "linearithmic" in s:
        return "nlogn"
    if "logn" in s or "logarithmic" in s:
        return "logn"
    if "linear" in s or "o(n)" in s:
        return "linear"
    if "constant" in s or "o(1)" in s:
        return "constant"
    return UNCLASSIFIED


def evaluate(y_true, y_pred_raw):
    """
    Compute metrics from ground-truth labels and raw model outputs.

    Returns a dict with accuracy, macro F1, the 7x7 confusion matrix (over
    classified predictions), per-class recall, and the number of unclassified
    outputs.
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    y_pred = [normalize_prediction(p) for p in y_pred_raw]

    # Unclassified predictions never match the truth, so they count as failures.
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

    # Confusion matrix over classified predictions only (thesis convention).
    classified = [(t, p) for t, p in zip(y_true, y_pred) if p in CLASSES]
    cm = confusion_matrix(
        [t for t, _ in classified],
        [p for _, p in classified],
        labels=CLASSES,
    )

    per_class_recall = {}
    for i, label in enumerate(CLASSES):
        row = cm[i].sum()
        per_class_recall[label] = cm[i, i] / row if row else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_recall": per_class_recall,
        "n_unclassified": sum(1 for p in y_pred if p not in CLASSES),
        "n_total": len(y_true),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, title="Confusion matrix", save_path=None):
    """Plot a confusion matrix in the style of the thesis figures."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [DISPLAY[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # Show once and close: returning the figure would make the notebook render it a
    # second time (on top of matplotlib's inline auto-display).
    plt.show()
    plt.close(fig)


def print_summary(name, results):
    """Print a one-line summary plus per-class recall."""
    print(f"[{name}] accuracy={results['accuracy']:.3f}  "
          f"macro_f1={results['macro_f1']:.3f}  "
          f"unclassified={results['n_unclassified']}/{results['n_total']}")
    for label in CLASSES:
        print(f"    {label:<12} recall={results['per_class_recall'][label]:.3f}")

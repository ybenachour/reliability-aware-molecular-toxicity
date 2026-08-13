from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def save_reliability_diagram(y_true, probability, path: str | Path, title: str = "Reliability diagram") -> Path:
    fraction, mean = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(mean, fraction, marker="o", label="Model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction", title=title)
    ax.legend()
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_confusion_matrix(matrix, path: str | Path, title: str = "Confusion matrix") -> Path:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(values)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

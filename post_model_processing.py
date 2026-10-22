import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from train_utils import RESULTS_DIR


def extract_features(model, dataloader, device):
    """Activations of the model's last non-linear layer for every sample, with their labels."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for spectrogram, lengths, target in dataloader:
            features.append(model.features(spectrogram.to(device), lengths).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)


def t_sne_plot(features, labels, name):
    """2D t-SNE projection of the features, coloured by digit, saved to results/<name>_tsne.png"""
    features_tsne = TSNE(n_components=2, random_state=0).fit_transform(features)
    plt.figure(figsize=(8, 8))
    for i in range(10):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=str(i), s=8)
    plt.legend(title="Digit")
    plt.title(f"t-SNE: {name}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_tsne.png"), bbox_inches="tight")
    plt.close()


def t_sne_evaluation(model, dataloader, device, name):
    t_sne_plot(*extract_features(model, dataloader, device), name)

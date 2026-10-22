"""Training and evaluation code shared by the task scripts."""
import copy
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from preprocessor import pad_collate

RESULTS_DIR = "results"


def set_seed(seed):
    """Seed every random number generator used, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_metadata(path="SDR_metadata.tsv"):
    return pd.read_csv(path, sep="\t", header=0, index_col="Unnamed: 0")


def split_data(audio_df, speaker=None):
    """Split the data into a train, valid and test set, optionally training on one speaker only"""
    training_df = audio_df.loc[audio_df["split"] == "TRAIN"]
    if speaker:
        assert speaker in set(training_df.speaker), f"{speaker} is not a training speaker"
        training_df = training_df.loc[training_df["speaker"] == speaker]

    testing_df = audio_df.loc[audio_df["split"] == "TEST"]
    val_df = audio_df.loc[audio_df["split"] == "DEV"]

    print(f"# Train Size: {len(training_df)}")
    print(f"# Valid Size: {len(val_df)}")
    print(f"# Test Size: {len(testing_df)}")

    return training_df, val_df, testing_df


def make_loader(dataset, batch_size=32, shuffle=False, collate_fn=pad_collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def predict(model, data_loader, device):
    """Return (targets, predictions) of a model on a data set."""
    model.eval()
    y_target, y_prediction = [], []
    with torch.no_grad():
        for features, lengths, target in data_loader:
            output = model(features.to(device), lengths)
            y_prediction.append(output.argmax(dim=1).cpu())
            y_target.append(target)
    return torch.cat(y_target).numpy(), torch.cat(y_prediction).numpy()


def evaluate(model, data_loader, device):
    """Accuracy (%) of a model on a data set."""
    y_target, y_prediction = predict(model, data_loader, device)
    return 100.0 * (y_target == y_prediction).mean()


def train(model, train_loader, validation_loader, optimizer, device, num_epochs=50, patience=10):
    """
    Train with cross-entropy, evaluating on the validation set after every epoch.
    Stops when validation accuracy has not improved for `patience` epochs and
    restores the weights of the best epoch. Returns the validation accuracy per epoch.
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    accs = []
    best_acc, best_state = -1.0, None
    # Exponential moving average of the loss.
    ema_loss = None

    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        for features, lengths, target in train_loader:
            # Forward pass.
            output = model(features.to(device), lengths)
            loss = criterion(output, target.to(device))

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_loss = loss.item() if ema_loss is None else ema_loss + (loss.item() - ema_loss) * 0.1

        acc = evaluate(model, validation_loader, device)
        accs.append(acc)
        print(
            f"Epoch: {epoch + 1}\tLoss: {ema_loss:.4f}\tVal acc: {acc:.1f}%\tTime: {time.time() - tick:.1f}s"
        )

        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        elif epoch - int(np.argmax(accs)) >= patience:
            print(f"Early stopping, best val acc {best_acc:.1f}% at epoch {int(np.argmax(accs)) + 1}")
            break

    model.load_state_dict(best_state)
    return accs


def report(name, model, loaders, device):
    """
    Print and save accuracy, confusion matrix and classification report of a model
    on each of the given {split_name: loader}, and record the accuracies in results.json.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    accuracies = {}
    lines = []
    for split, loader in loaders.items():
        y_target, y_prediction = predict(model, loader, device)
        accuracies[split] = round(100.0 * float((y_target == y_prediction).mean()), 2)
        lines += [
            f"----- {name} | {split} -----",
            f"Accuracy: {accuracies[split]:.2f}%",
            f"Confusion matrix:\n{confusion_matrix(y_target, y_prediction)}",
            classification_report(y_target, y_prediction, digits=3),
        ]
        ConfusionMatrixDisplay.from_predictions(y_target, y_prediction, colorbar=False)
        plt.title(f"{name} ({split}, acc {accuracies[split]:.1f}%)")
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_{split}_confusion.png"), bbox_inches="tight")
        plt.close()

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS_DIR, f"{name}_report.txt"), "w") as f:
        f.write(text)
    save_result(name, accuracies)
    return accuracies


def save_result(name, accuracies):
    """Merge one experiment's accuracies into results/results.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "results.json")
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results[name] = accuracies
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def plot_curve(name, accs):
    plt.figure()
    plt.plot(range(1, len(accs) + 1), accs)
    plt.title(f"Validation accuracy: {name}")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_val_curve.png"), bbox_inches="tight")
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

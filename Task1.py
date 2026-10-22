"""Task I: linear baseline (SGDClassifier) on downsampled mel spectrograms."""
import argparse

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessor import downsample_spectrogram, load_spectrograms
from train_utils import load_metadata, save_result, set_seed, split_data


def downsampled_features(df, n):
    """Stack the downsampled (n * K) vectors of every clip in df."""
    spectrograms = load_spectrograms(df)
    return np.array([downsample_spectrogram(spectrograms[i], n) for i in df.identifier])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of time splits (N <= 25)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_df, valid_df, test_df = split_data(load_metadata())

    clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=args.seed)
    clf.fit(downsampled_features(train_df, args.n), train_df.label)

    accuracies = {}
    for split, df in (("dev", valid_df), ("test", test_df)):
        y_pred = clf.predict(downsampled_features(df, args.n))
        accuracies[split] = round(100.0 * accuracy_score(df.label, y_pred), 2)
        print(f"----- Baseline | {split} -----")
        print(f"Accuracy: {accuracies[split]:.2f}%")
        print(f"Confusion matrix:\n{confusion_matrix(df.label, y_pred)}")
        print(classification_report(df.label, y_pred, digits=3))

    save_result(f"task1_linear_n{args.n}", accuracies)

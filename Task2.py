"""Task II: CNN and LSTM on the full, variable-length mel spectrograms, plus t-SNE analysis."""
import argparse

import torch

from models import CNN, RNNModel
from post_model_processing import t_sne_evaluation, t_sne_plot
from preprocessor import SpectrogramDataset
from Task1 import downsampled_features
from train_utils import (
    count_parameters,
    get_device,
    load_metadata,
    make_loader,
    plot_curve,
    report,
    set_seed,
    split_data,
    train,
)

MODELS = {"cnn": CNN, "rnn": RNNModel}


def run(model_name, train_df, valid_df, test_df, device, args):
    """Train one model on the full training set, then report and visualise it."""
    set_seed(args.seed)
    name = f"task2_{model_name}"
    print(f"===== {name} =====")

    train_loader = make_loader(SpectrogramDataset(train_df), args.batch_size, shuffle=True)
    valid_loader = make_loader(SpectrogramDataset(valid_df), args.batch_size)
    test_loader = make_loader(SpectrogramDataset(test_df), args.batch_size)

    model = MODELS[model_name]()
    print("Num Parameters:", count_parameters(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    accs = train(model, train_loader, valid_loader, optimizer, device, args.epochs, args.patience)
    plot_curve(name, accs)
    report(name, model, {"dev": valid_loader, "test": test_loader}, device)
    t_sne_evaluation(model, test_loader, device, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device used: {device}")
    train_df, valid_df, test_df = split_data(load_metadata())

    # t-SNE of the downsampled baseline representation, to compare with the neural models
    t_sne_plot(downsampled_features(test_df, 10), test_df.label.values, "task2_downsampled_baseline")

    for model_name in args.models:
        run(model_name, train_df, valid_df, test_df, device, args)

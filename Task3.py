"""
Task III: train on a single speaker, with data augmentation (SpecAugment on the
spectrogram, pitch shifting on the waveform) and contrastive pre-training.
"""
import argparse
import time

import torch

from models import CNN, ContrastiveModel, RNNModel
from post_model_processing import t_sne_evaluation
from preprocessor import ContrastiveLoss, SpectrogramDataset, TwoViewDataset, two_view_collate
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

# experiment name -> (model class, augmentation)
SUPERVISED_EXPERIMENTS = {
    "cnn": (CNN, None),
    "rnn": (RNNModel, None),
    "cnn_specaugment": (CNN, "specaugment"),
    "cnn_pitch": (CNN, "pitch"),
    "cnn_both": (CNN, "both"),
}
EXPERIMENTS = list(SUPERVISED_EXPERIMENTS) + ["cnn_contrastive"]


def fit_and_report(name, model, train_loader, eval_loaders, device, args, params=None):
    """Train `model` (or only `params` of it) with cross-entropy, then report and save its results."""
    params = model.parameters() if params is None else params
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    accs = train(model, train_loader, eval_loaders["dev"], optimizer, device, args.epochs, args.patience)
    plot_curve(name, accs)
    report(name, model, eval_loaders, device)


def contrastive_pretrain(model, train_loader, device, args):
    """
    Pre-train with the margin-based contrastive loss on two augmented views of each clip.
    Positive pairs are the two views of one clip ("views") or all clips of the same digit ("labels").
    """
    model.to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.pretrain_epochs):
        tick = time.time()
        model.train()
        total_loss = 0.0
        for features, lengths, labels, indices in train_loader:
            embeddings = model(features.to(device), lengths)
            group_ids = indices if args.positives == "views" else labels
            loss = criterion(embeddings, group_ids.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Pre-train epoch: {epoch + 1}\tLoss: {total_loss / len(train_loader):.4f}\tTime: {time.time() - tick:.1f}s"
        )


def run_contrastive(name, train_df, eval_loaders, device, args):
    """Contrastive pre-training of a CNN encoder, then a linear probe and full fine-tuning."""
    two_views = TwoViewDataset(SpectrogramDataset(train_df, augment="both"))
    pretrain_loader = make_loader(two_views, args.batch_size, shuffle=True, collate_fn=two_view_collate)

    encoder = CNN()
    contrastive_model = ContrastiveModel(encoder)
    print("Num Parameters:", count_parameters(contrastive_model))
    contrastive_pretrain(contrastive_model, pretrain_loader, device, args)
    t_sne_evaluation(encoder, eval_loaders["test"], device, f"{name}_pretrained")

    # linear probe: only the final classification layer is trained on the frozen encoder
    plain_loader = make_loader(SpectrogramDataset(train_df), args.batch_size, shuffle=True)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.classifier.weight.requires_grad = True
    encoder.classifier.bias.requires_grad = True
    fit_and_report(
        f"{name}_linear_probe", encoder, plain_loader, eval_loaders, device, args,
        params=encoder.classifier.parameters(),
    )

    # fine-tuning: the whole network is trained, with the same augmentation as "cnn_both"
    for param in encoder.parameters():
        param.requires_grad = True
    augmented_loader = make_loader(SpectrogramDataset(train_df, augment="both"), args.batch_size, shuffle=True)
    fit_and_report(f"{name}_finetune", encoder, augmented_loader, eval_loaders, device, args)
    t_sne_evaluation(encoder, eval_loaders["test"], device, f"{name}_finetune")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="george", help="the only training speaker")
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--positives", default="views", choices=["views", "labels"],
                        help="contrastive positives: views of the same clip, or clips of the same digit")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device used: {device}")
    print("Train and Test Split based on speaker!")
    train_df, valid_df, test_df = split_data(load_metadata(), speaker=args.speaker)
    eval_loaders = {
        "dev": make_loader(SpectrogramDataset(valid_df), args.batch_size),
        "test": make_loader(SpectrogramDataset(test_df), args.batch_size),
    }

    for experiment in args.experiments:
        set_seed(args.seed)
        name = f"task3_{args.speaker}_{experiment}"
        print(f"===== {name} =====")
        if experiment == "cnn_contrastive":
            run_contrastive(f"{name}_{args.positives}", train_df, eval_loaders, device, args)
            continue

        model_class, augment = SUPERVISED_EXPERIMENTS[experiment]
        model = model_class()
        print("Num Parameters:", count_parameters(model))
        train_loader = make_loader(SpectrogramDataset(train_df, augment=augment), args.batch_size, shuffle=True)
        fit_and_report(name, model, train_loader, eval_loaders, device, args)
        t_sne_evaluation(model, eval_loaders["test"], device, name)

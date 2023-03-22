import time

import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CNN, RNNModel, CNNContrastive
from post_model_processing import t_sne_evaluation
from preprocessor import SpectrogramDataset, ContrastiveLoss
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = y_true.float()
    loss_contrastive = torch.mean(
        (1 - y_true) * torch.square(y_pred)
        + y_true * torch.square(torch.clamp(margin - y_pred, min=0.0))
    )
    return loss_contrastive


def test(model, data_loader, verbose=False, verbose_report=False):
    """Measures the accuracy of a model on a data set."""
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    y_prediction = None
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(
            data_loader, total=len(data_loader.batch_sampler), desc="Testing"
        ):
            # Forward pass.
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            if y_prediction is None:
                y_prediction = pred.squeeze()
                y_target = target
            else:
                y_prediction = torch.cat((y_prediction, pred.squeeze()), dim=0)
                y_target = torch.cat((y_target, target), dim=0)

            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100.0 * correct / len(data_loader.sampler)
    if verbose:
        print("----- Model Evaluation -----")
        print(f"Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)")
    if verbose_report:
        y_target = y_target.cpu()
        y_prediction = y_prediction.cpu()
        cm_vr = confusion_matrix(y_target, y_prediction)
        accuracy_vr = accuracy_score(y_target, y_prediction)
        report_vr = classification_report(y_target, y_prediction)
        print(f"The Confusion matrix Test set:\n{cm_vr}")
        print(f"The Accuracy for Test set:\n{accuracy_vr}")
        print(f"The Report for Test set:\n{report_vr}")

    return percent


# using glorot initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def train(model, criterion, train_loader, validation_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model."""

    # Move model to the device (CPU or GPU).
    model.to(device)

    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

    #     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(
            enumerate(train_loader),
            total=len(train_loader.batch_sampler),
            desc="Training",
        ):
            # Forward pass.
            output = model(features.to(device))
            loss = criterion(output.to(device), target.to(device))

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01

        tock = time.time()
        acc = test(model, validation_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print(
            "Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds".format(
                epoch + 1, ema_loss, tock - tick
            ),
        )
    return accs


def train_w_cl(model, criterion, train_loader, validation_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model."""

    (cl_train_loader, actual_train_loader) = train_loader

    # Move model to the device (CPU or GPU).
    model.to(device)

    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

    #     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            cl_train_loader.sampler.shuffle = False
            actual_train_loader.sampler.shuffle = False
        else:
            cl_train_loader.sampler.shuffle = True
            actual_train_loader.sampler.shuffle = True

        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, ((features1, target1), (features2, target2)) in tqdm(
            enumerate(zip(cl_train_loader, actual_train_loader)),
            total=len(actual_train_loader),
            desc="Training",
        ):
            # Forward pass.
            output1, output2 = model(features1.to(device), features2.to(device))
            loss = criterion(output1.to(device), output2.to(device), target1.to(device), target2.to(device))

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01

        tock = time.time()
        acc = test(model, validation_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print(
            "Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds".format(
                epoch + 1, ema_loss, tock - tick
            ),
        )
    return accs


def split_data(audio_df, speaker=None):
    """Split the data into a train, valid and test set"""
    if speaker and speaker in set(audio_df.speaker.values):
        training_df = audio_df.loc[
            (audio_df["split"] == "TRAIN") & (audio_df["speaker"] == speaker)
        ]
    else:
        training_df = audio_df.loc[audio_df["split"] == "TRAIN"]

    testing_df = audio_df.loc[audio_df["split"] == "TEST"]
    val_df = audio_df.loc[audio_df["split"] == "DEV"]

    print(f"# Train Size: {len(training_df)}")
    print(f"# Valid Size: {len(val_df)}")
    print(f"# Test Size: {len(testing_df)}")

    return training_df, val_df, testing_df


def build_training_data(
    train_df,
    valid_df,
    test_df,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    n=0,
    flattened=True,
    data_augmentation=False,
    use_contrastive_loss=False,
    shuffle=True,
):
    """Covert the audio samples into training data"""

    train_data = SpectrogramDataset(train_df, n=n, flattened=flattened, data_augmentation=data_augmentation, use_contrastive_loss=use_contrastive_loss)
    train_pr = DataLoader(
        train_data, batch_size=train_batch_size, shuffle=shuffle, num_workers=2
    )

    valid_data = SpectrogramDataset(valid_df, n=n, flattened=flattened)
    valid_pr = DataLoader(
        valid_data, batch_size=val_batch_size, shuffle=shuffle, num_workers=2
    )

    test_data = SpectrogramDataset(test_df, n=n, flattened=flattened)
    test_pr = DataLoader(
        test_data, batch_size=test_batch_size, shuffle=shuffle, num_workers=2
    )

    return train_pr, valid_pr, test_pr


def start(cnn=False, use_contrastive_loss=False, data_augmentation=False):
    if cnn and not use_contrastive_loss:
        # normalize data with n=15 for Deep CNN model
        print("Preparing Data for Deep CNN!")
        train_loader, valid_loader, test_loader = build_training_data(
            speaker_train_df, speaker_valid_df, speaker_test_df, 32, 32, 32, n=15, data_augmentation=data_augmentation
        )

        CnnModel = CNN()

        print("Num Parameters:", sum([p.numel() for p in CnnModel.parameters()]))
        CnnModel.apply(init_weights)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(CnnModel.parameters(), weight_decay=1e-4)

        num_epochs = 100
        accs = train(
            CnnModel,
            criterion,
            train_loader,
            valid_loader,
            optimizer,
            num_epochs=num_epochs,
        )
        test(CnnModel, test_loader, verbose=True, verbose_report=True)

        plt.plot(accs)
        plt.title("Validation Accuracy CNN")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy (%)")
        plt.show()

    elif not cnn:

        print("Preparing Data for Audio RNN LSTM!")
        at_train_loader, at_valid_loader, at_test_loader = build_training_data(
            speaker_train_df,
            speaker_valid_df,
            speaker_test_df,
            32,
            32,
            32,
            15,
            flattened=False,
            data_augmentation=data_augmentation
        )

        AudioRNNModel = RNNModel()
        print("Num Parameters:", sum([p.numel() for p in AudioRNNModel.parameters()]))
        ARCriterion = torch.nn.CrossEntropyLoss()
        AROptimizer = torch.optim.Adam(AudioRNNModel.parameters(), weight_decay=1e-4)

        ARaccs = train(
            AudioRNNModel,
            ARCriterion,
            at_train_loader,
            at_valid_loader,
            AROptimizer,
            num_epochs=100,
        )

        test(AudioRNNModel, at_test_loader, verbose=True, verbose_report=True)

        plt.plot(ARaccs)
        plt.title("Validation Accuracy RNN")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy (%)")
        plt.show()

        t_sne_evaluation(AudioRNNModel, at_test_loader, device)

    elif cnn and use_contrastive_loss:
        print("Preparing Data for Audio RNN LSTM!")
        cl_train_loader, cl_valid_loader, cl_test_loader = build_training_data(speaker_train_df, speaker_valid_df, speaker_test_df, 32, 32, 32, 15, data_augmentation=True, use_contrastive_loss=True)
        train_loader, valid_loader, test_loader = build_training_data(speaker_train_df, speaker_valid_df, speaker_test_df, 32, 32, 32, 15)

        ContrastiveCNNModel = CNNContrastive()
        print("Num Parameters:", sum([p.numel() for p in ContrastiveCNNModel.parameters()]))
        cl_criterion = ContrastiveLoss()
        cl_optimizer = torch.optim.Adam(ContrastiveCNNModel.parameters(), weight_decay=1e-4)

        ARaccs = train_w_cl(
            ContrastiveCNNModel,
            cl_criterion,
            (cl_train_loader, train_loader),
            cl_valid_loader,
            cl_optimizer,
            num_epochs=100,
        )

        test(ContrastiveCNNModel, test_loader, verbose=True, verbose_report=True)

        plt.plot(ARaccs)
        plt.title("Validation Accuracy RNN")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy (%)")
        plt.show()


if __name__ == "__main__":

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    sdr_df = pd.read_csv("SDR_metadata.tsv", sep="\t", header=0, index_col="Unnamed: 0")

    print("Train and Test Split based on speaker!")
    speaker_train_df, speaker_valid_df, speaker_test_df = split_data(
        sdr_df, speaker="george"
    )

    # start(cnn=True) # Training Done
    # start(cnn=False) # Training Done
    # start(cnn=True, data_augmentation=True)
    start(cnn=False, data_augmentation=True)
    # start(cnn=True, use_contrastive_loss=True) # Training Done

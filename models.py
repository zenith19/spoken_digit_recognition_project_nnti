import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def length_mask(x, lengths):
    """(B, 1, T) mask of a (B, C, T) tensor that is 1 on real frames and 0 on padding."""
    mask = torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None].to(x.device)
    return mask.unsqueeze(1).float()


def masked_mean(x, lengths):
    """Mean over the time axis of a (B, C, T) tensor, ignoring padded frames."""
    mask = length_mask(x, lengths)
    return (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)


class ConvBlock(nn.Module):
    """Two 1D convolutions over time (with batch norm and ReLU) followed by optional max pooling."""

    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = pool

    def forward(self, x, lengths):
        # padded frames are reset to 0 after each convolution so they never leak into real frames
        x = x * length_mask(x, lengths)
        x = F.relu(self.bn1(self.conv1(x))) * length_mask(x, lengths)
        x = F.relu(self.bn2(self.conv2(x))) * length_mask(x, lengths)
        if self.pool:
            x = F.max_pool1d(x, kernel_size=2)
            lengths = lengths // 2
        return x, lengths


class CNN(nn.Module):
    """
    1D temporal CNN over a variable-length mel spectrogram: the mel bins are the
    input channels and convolutions slide over time. A masked global average pool
    turns any length into a fixed-size vector, so no downsampling is needed.
    """

    def __init__(self, num_mels=13, num_classes=10, hidden_size=128, dropout=0.3):
        super(CNN, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(num_mels, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256, pool=False),
            ]
        )
        self.fc1 = nn.Linear(256, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def features(self, x, lengths):
        """Output of the last non-linear layer, shape (B, hidden_size)."""
        x = x.transpose(1, 2)  # (B, T, K) -> (B, K, T)
        for block in self.blocks:
            x, lengths = block(x, lengths)
        x = masked_mean(x, lengths)
        return F.relu(self.fc1(x))

    def forward(self, x, lengths):
        return self.classifier(self.dropout(self.features(x, lengths)))


class RNNModel(nn.Module):
    """Bidirectional LSTM over a variable-length mel spectrogram, using packed sequences."""

    def __init__(self, num_mels=13, num_classes=10, rnn_size=128, hidden_size=128, dropout=0.3):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=num_mels,
            hidden_size=rnn_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(2 * rnn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def features(self, x, lengths):
        """Output of the last non-linear layer, shape (B, hidden_size)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        # final forward and backward states of the top layer
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return F.relu(self.fc1(x))

    def forward(self, x, lengths):
        return self.classifier(self.dropout(self.features(x, lengths)))


class ContrastiveModel(nn.Module):
    """
    Wraps an encoder (e.g. CNN) with a projection head that outputs L2-normalised
    embeddings for the contrastive loss (as in SimCLR, Chen et al., 2020). After
    pre-training, the encoder is used on its own for classification.
    """

    def __init__(self, encoder, projection_size=64):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        hidden_size = encoder.classifier.in_features
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x, lengths):
        return F.normalize(self.projection(self.encoder.features(x, lengths)), dim=1)

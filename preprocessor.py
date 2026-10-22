import os
import pickle
import warnings

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

SAMPLING_RATE = 8000  # This value is determined by the wav file, DO NOT CHANGE
NUM_MELS = 13
PITCH_STEPS = (-4, -3, -2, -1, 1, 2, 3, 4)  # semitones used for pitch-shift augmentation
CACHE_DIR = ".cache"
warnings.filterwarnings("ignore")


def extract_melspectrogram(signal, sr, num_mels):
    """
    Given a time series speech signal (.wav), sampling rate (sr),
    and the number of mel coefficients, return a mel-scaled
    representation of the signal as numpy array of shape (num_mels, T).
    """

    mel_features = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=200,  # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80,  # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels,  # number of frequency bins, use either 13 or 39
        fmin=50,  # min frequency threshold
        fmax=4000,  # max frequency threshold, set to SAMPLING_RATE/2
    )

    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20 * np.log10(mel_features)

    # per-utterance feature scaling of every mel band (zero mean, unit variance over time)
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)

    return scaled_log_mel_features.astype(np.float32)


def load_audio(file_path):
    """Load a wav file at the native 8 kHz sampling rate."""
    signal, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    return signal


def get_mel_spectrogram(file_path, num_mels=NUM_MELS, pitch_step=0):
    """Mel spectrogram of a wav file, optionally after shifting its pitch by `pitch_step` semitones."""
    signal = load_audio(file_path)
    if pitch_step:
        signal = librosa.effects.pitch_shift(signal, sr=SAMPLING_RATE, n_steps=pitch_step)
    return extract_melspectrogram(signal, SAMPLING_RATE, num_mels)


def load_spectrograms(df, num_mels=NUM_MELS, pitch_step=0):
    """
    Return {identifier: spectrogram} for every row of `df`.
    Spectrograms are cached on disk, so each file is only processed once.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"mel{num_mels}_pitch{pitch_step}.pkl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    missing = [row for row in df.itertuples() if row.identifier not in cache]
    if missing:
        for row in tqdm(missing, desc=f"Extracting features (pitch {pitch_step:+d})"):
            cache[row.identifier] = get_mel_spectrogram(row.file, num_mels, pitch_step)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return {identifier: cache[identifier] for identifier in df.identifier}


def downsample_spectrogram(spectrogram, n=15, flattened=True):
    """
    Given a spectrogram of an arbitrary length/duration (X ∈ K x T),
    return a downsampled version of the spectrogram v ∈ K * N
    """
    k, t = spectrogram.shape
    split_size = t // n
    split_indices = [i * split_size for i in range(n)] + [t]

    X_splits = [
        spectrogram[:, split_indices[i] : split_indices[i + 1]] for i in range(n)
    ]
    X_downsampled = np.array([np.mean(split, axis=1) for split in X_splits])

    if flattened:
        return X_downsampled.reshape((n * k,))
    return X_downsampled


def spec_augment(spectrogram, freq_mask_param=3, time_mask_ratio=0.1, num_masks=2):
    """
    SpecAugment-style masking (Park et al., 2019) of a (K, T) spectrogram.
    Masked regions are set to 0, i.e. the per-band mean after feature scaling.
    """
    masked = spectrogram.copy()
    k, t = masked.shape
    max_time_mask = max(1, int(t * time_mask_ratio))

    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        f0 = np.random.randint(0, k - f + 1)
        masked[f0 : f0 + f, :] = 0

        w = np.random.randint(0, max_time_mask + 1)
        t0 = np.random.randint(0, t - w + 1)
        masked[:, t0 : t0 + w] = 0

    return masked


class SpectrogramDataset(Dataset):
    """
    Mel spectrograms and labels of the clips in a dataframe.

    Each item is (features, label). With n=0 the features are the full,
    variable-length spectrogram of shape (T, K); with 1 <= n <= 25 they are
    downsampled to n time steps (flattened to n * K if `flattened`).

    `augment` is applied on the fly in __getitem__, so every epoch sees
    different views: None, "specaugment", "pitch" or "both".
    """

    def __init__(self, df, n=0, flattened=True, augment=None):
        assert augment in (None, "specaugment", "pitch", "both")
        self.n = max(0, min(n, 25))
        self.flattened = flattened
        self.augment = augment
        self.identifiers = list(df.identifier)
        self.labels = list(df.label)
        self.spectrograms = load_spectrograms(df)

        # pitch-shifted versions are precomputed once, then one is drawn at random per item
        self.pitch_shifted = []
        if augment in ("pitch", "both"):
            self.pitch_shifted = [load_spectrograms(df, pitch_step=step) for step in PITCH_STEPS]

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        spectrogram = self.spectrograms[identifier]

        if self.pitch_shifted and np.random.rand() < 0.5:
            spectrogram = self.pitch_shifted[np.random.randint(len(self.pitch_shifted))][identifier]
        if self.augment in ("specaugment", "both"):
            spectrogram = spec_augment(spectrogram)

        if self.n:
            features = downsample_spectrogram(spectrogram, self.n, self.flattened)
        else:
            features = spectrogram.T  # (T, K): time-major for the sequence models

        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


class TwoViewDataset(Dataset):
    """Returns two independently augmented views of the same clip, for contrastive learning."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        view1, label = self.dataset[idx]
        view2, _ = self.dataset[idx]
        return view1, view2, label, idx


def pad_collate(batch):
    """Zero-pad variable-length (T, K) spectrograms into a (B, T_max, K) batch."""
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])
    return pad_sequence(features, batch_first=True), lengths, torch.tensor(labels)


def two_view_collate(batch):
    """Collate (view1, view2, label, idx) items into one padded batch of 2B views."""
    view1, view2, labels, indices = zip(*batch)
    x, lengths, _ = pad_collate([(v, 0) for v in view1 + view2])
    return x, lengths, torch.tensor(labels * 2), torch.tensor(indices * 2)


class ContrastiveLoss(torch.nn.Module):
    """
    Margin-based contrastive loss (Hadsell et al., 2006) over all pairs in a batch
    of L2-normalised embeddings. Pairs with the same group id (two views of one
    clip, or the same label in the supervised variant) are pulled together; other
    pairs are pushed at least `margin` apart.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, group_ids):
        distances = torch.cdist(embeddings, embeddings)
        same = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
        not_self = ~torch.eye(len(group_ids), dtype=torch.bool, device=embeddings.device)

        positive = distances[same & not_self].square().mean()
        negative = F.relu(self.margin - distances[~same]).square().mean()
        return positive + negative

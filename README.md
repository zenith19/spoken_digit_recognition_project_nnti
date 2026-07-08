# Speaker-Independent Spoken Digit Recognition (xSDR)

Deep learning models for recognizing spoken digits (0–9) from short audio clips, with a focus on **speaker-independent generalization** — evaluating on speakers the model has never heard during training.

Built with PyTorch on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) (3,000 recordings, 6 speakers, 8 kHz). Course project for *Neural Networks: Theory and Implementation* (NNTI), Saarland University.

## Why speaker independence?

Real-world ASR systems must work for speakers they were never trained on. This is especially relevant for low-resource languages, where annotated speech may exist for only one or two speakers. This project simulates that constraint in two settings:

1. **Multi-speaker training** (`Task2.py`) — train on all speakers' TRAIN split, evaluate on held-out DEV/TEST splits.
2. **Single-speaker training** (`Task3.py`) — train on *one* speaker only ("george") and test generalization to the other five, using data augmentation and contrastive learning to close the gap.

## Approach

### Feature extraction (`preprocessor.py`)
- Audio → **log mel spectrogram** (13 mel bins, 25 ms window / 10 ms hop at 8 kHz), per-feature standardized
- **Fixed-length downsampling**: variable-length spectrograms are pooled into N=15 temporal segments (mean pooling), yielding a fixed 13×15 representation — flattened for the CNN, kept as a sequence for the LSTM

### Models (`models.py`)
| Model | Architecture | Input |
|---|---|---|
| `CNN` | 12-layer 1D ConvNet (64→128→256 channels) with BatchNorm, max pooling, Xavier init | Flattened spectrogram |
| `RNNModel` | Single-layer LSTM (hidden 256) → 2 FC layers | Spectrogram as a 13-dim feature sequence |
| `CNNContrastive` | Siamese variant of the CNN for pairwise contrastive training | Spectrogram pairs |

### Generalization techniques (`Task3.py`)
- **Frequency masking** (SpecAugment-style): random mel-bin bands zeroed out during training
- **Pitch shifting** (torchaudio): ±4 semitone perturbation of the waveform
- **Contrastive loss**: pulls same-digit embeddings together and pushes different-digit embeddings apart (margin-based, Euclidean distance), encouraging speaker-invariant representations

### Evaluation (`Task2.py`, `post_model_processing.py`)
- Accuracy, per-class precision/recall/F1, confusion matrix
- **t-SNE visualization** of learned embeddings to inspect class separation across speakers

## Project structure

```
├── Task2.py                   # Multi-speaker training: CNN + LSTM baselines
├── Task3.py                   # Single-speaker training + augmentation + contrastive learning
├── models.py                  # CNN, LSTM, and Siamese CNN architectures
├── preprocessor.py            # Mel spectrogram extraction, downsampling, augmentation, Dataset class
├── post_model_processing.py   # t-SNE embedding visualization
├── SDR_metadata.tsv           # File paths, speaker IDs, labels, TRAIN/DEV/TEST split
└── speech_data/               # 3,000 WAV files (digit_speaker_index.wav)
```

## Getting started

```bash
pip install -r requirements.txt
pip install torch torchaudio matplotlib   # not pinned in requirements.txt

# Multi-speaker setting: trains CNN, then LSTM (100 epochs each)
python Task2.py

# Single-speaker setting: CNN baseline → + augmentation → + contrastive loss
python Task3.py
```

Both scripts print per-epoch validation accuracy and a final test report (accuracy, classification report, confusion matrix), then display validation-accuracy curves and t-SNE plots.

## Dataset

Free Spoken Digit Dataset (FSDD): 6 speakers × 10 digits × 50 recordings = 3,000 WAV files at 8 kHz. The split in `SDR_metadata.tsv` keeps DEV and TEST speakers disjoint from training where the speaker-independent setting requires it.

## Notes & known limitations

- Trained on CPU or a single GPU; the dataset is small enough that full runs finish quickly
- `requirements.txt` predates the PyTorch port and omits `torch`, `torchaudio`, and `matplotlib`
- The pitch-shifting augmentation path in `SpectrogramDataset` is currently disabled (dead branch); frequency masking is the active augmentation
- Hyperparameters (100 epochs, Adam with weight decay 1e-4, batch size 32) are fixed in the scripts rather than exposed as CLI arguments

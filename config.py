"""
Lipika Project Configuration.
Defines hyperparameters, directory paths, model configurations, and device settings.
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FONTS_DIR = BASE_DIR / "fonts"

for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR, FONTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset specifications matching the IEEE paper
NUM_CLASSES = 47          # 13 Swaras (Vowels) + 34 Vyanjanas (Consonants)
NUM_VOWELS = 13
NUM_CONSONANTS = 34
IMAGE_SIZE = (28, 28)     # Standard HCR image resolution
CHANNELS = 1              # Grayscale

# Hardware Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0           # Safe for Windows multi-processing

# Training Hyperparameters
BATCH_SIZE = 64
BASELINE_EPOCHS = 50      # Baseline as described in paper
FAST_EPOCHS = 20          # Fast-KannadaNet with OneCycleLR reaches higher accuracy in 20 epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# Fast Hierarchical Routing Weight
ROUTING_LOSS_WEIGHT = 0.3  # Multi-task auxiliary loss for Vowel/Consonant separation

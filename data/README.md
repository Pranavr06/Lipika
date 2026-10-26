# Lipika Dataset Directory

This directory manages training and evaluation datasets for Kannada Handwritten Character Recognition:

- `kannada_varnamale.py`: Linguistic character definitions and Unicode mappings for all 47 classes (13 Swaras + 34 Vyanjanas).
- `dataset_generator.py`: Realistic handwriting synthesizer simulating stroke variations, elastic distortions, and pen dynamics.
- `dataset_loader.py`: High-speed vectorized PyTorch DataLoader with zero-copy caching.
- `raw/`: Directory for uncompressed raw scans and external datasets (e.g. Chars74k Kannada handwritten character dataset).
- `processed/`: Directory for compiled compressed `.npz` arrays (`kannada_dataset_28x28.npz`).
- `downloads/`: Directory for downloaded dataset archives.

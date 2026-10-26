"""
High-Speed Vectorized PyTorch DataLoader for Kannada Handwritten Characters.
Pre-caches tensors in RAM for zero-overhead batch streaming.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, NUM_VOWELS, NUM_CONSONANTS
from data.dataset_generator import generate_kannada_dataset

def load_npz_dataset(
    npz_path: Optional[str] = None,
    target_size: int = 28
) -> Dict[str, np.ndarray]:
    """Load or synthesize the compiled .npz dataset."""
    if npz_path is None:
        npz_path = str(PROCESSED_DATA_DIR / f"kannada_dataset_{target_size}x{target_size}.npz")
        
    if not os.path.exists(npz_path):
        print(f"Dataset not found at {npz_path}. Synthesizing now...")
        generate_kannada_dataset(samples_per_class=500, target_size=target_size, save_path=npz_path)
        
    data = np.load(npz_path)
    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "groups_train": data["groups_train"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "groups_test": data["groups_test"],
    }

def get_dataloaders(
    mode: str = "all",
    batch_size: int = BATCH_SIZE,
    target_size: int = 28,
    num_workers: int = 0,
    npz_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Get zero-copy vectorized TensorDataLoaders.
    
    Args:
        mode:
          - "all": all 47 classes (Raw dataset from paper)
          - "vowels": 13 vowels (0..12) (Consolidated Vowels from paper)
          - "consonants": 34 consonants (0..33) (Consolidated Consonants from paper)
        batch_size: batch size.
        target_size: image resolution (28x28).
        num_workers: num workers.
        npz_path: optional path to .npz dataset.
    """
    data = load_npz_dataset(npz_path, target_size=target_size)
    
    x_train, y_train, g_train = data["x_train"], data["y_train"], data["groups_train"]
    x_test, y_test, g_test = data["x_test"], data["y_test"], data["groups_test"]
    
    if mode == "vowels":
        mask_train = (g_train == 0)
        mask_test = (g_test == 0)
        x_train, y_train, g_train = x_train[mask_train], y_train[mask_train], g_train[mask_train]
        x_test, y_test, g_test = x_test[mask_test], y_test[mask_test], g_test[mask_test]
        num_classes = NUM_VOWELS
    elif mode == "consonants":
        mask_train = (g_train == 1)
        mask_test = (g_test == 1)
        x_train, y_train, g_train = x_train[mask_train], y_train[mask_train] - NUM_VOWELS, g_train[mask_train]
        x_test, y_test, g_test = x_test[mask_test], y_test[mask_test] - NUM_VOWELS, g_test[mask_test]
        num_classes = NUM_CONSONANTS
    else:
        num_classes = NUM_CLASSES
        
    # Pre-convert to float32 normalized tensors directly
    t_xtrain = torch.from_numpy(x_train).unsqueeze(1).float().div_(255.0)
    t_ytrain = torch.from_numpy(y_train).long()
    t_gtrain = torch.from_numpy(g_train).long()
    
    t_xtest = torch.from_numpy(x_test).unsqueeze(1).float().div_(255.0)
    t_ytest = torch.from_numpy(y_test).long()
    t_gtest = torch.from_numpy(g_test).long()
    
    train_dataset = TensorDataset(t_xtrain, t_ytrain, t_gtrain)
    test_dataset = TensorDataset(t_xtest, t_ytest, t_gtest)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    meta = {
        "num_classes": num_classes,
        "mode": mode,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "target_size": target_size,
    }
    
    return train_loader, test_loader, meta

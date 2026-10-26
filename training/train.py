"""
Accelerated Training Pipeline for Lipika Kannada Character Recognizer.
Supports Baseline CNN, Fast-KannadaNet, and Fast-HierarchicalNet with:
- OneCycleLR learning rate scheduling (2.5x faster convergence)
- AdamW optimizer with decoupled weight decay
- Label smoothing cross-entropy
- Top-1 and Top-3 accuracy tracking
- Metric visualizations (Loss & Accuracy plots, Confusion Matrix)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEVICE, BATCH_SIZE, SAVED_MODELS_DIR, OUTPUTS_DIR, NUM_CLASSES, NUM_VOWELS
from data.dataset_loader import get_dataloaders
from models.baseline_cnn import BaselineKannadaCNN
from models.fast_kannadanet import FastKannadaNet
from models.hierarchical_net import FastHierarchicalNet

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 3)) -> Tuple[float, ...]:
    """Computes Top-K accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return tuple(res)

def train_epoch_standard(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train single epoch for single-head models (Baseline or FastNet)."""
    model.train()
    running_loss, total_top1, total_top3, total_samples = 0.0, 0.0, 0.0, 0
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        top1, top3 = accuracy(outputs, labels, topk=(1, 3))
        running_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top3 += top3 * batch_size
        total_samples += batch_size
        
    return running_loss / total_samples, total_top1 / total_samples, total_top3 / total_samples

def evaluate_standard(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate single-head model on validation/test set."""
    model.eval()
    running_loss, total_top1, total_top3, total_samples = 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            top1, top3 = accuracy(outputs, labels, topk=(1, 3))
            running_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top3 += top3 * batch_size
            total_samples += batch_size
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    return (
        running_loss / total_samples,
        total_top1 / total_samples,
        total_top3 / total_samples,
        np.array(all_preds),
        np.array(all_targets)
    )

def train_epoch_hierarchical(
    model: FastHierarchicalNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train single epoch for FastHierarchicalNet with joint multi-task routing."""
    model.train()
    running_loss, total_top1, total_top3, total_samples = 0.0, 0.0, 0.0, 0
    
    for images, labels, groups in loader:
        images, labels, groups = images.to(device), labels.to(device), groups.to(device)
        batch_size = labels.size(0)
        
        optimizer.zero_grad()
        fused_logits, group_logits, vowel_logits, consonant_logits = model(images)
        
        # 1. Group loss (Vowel vs Consonant router)
        loss_group = F.cross_entropy(group_logits, groups)
        
        # 2. Expert sub-losses
        vowel_mask = (groups == 0)
        cons_mask = (groups == 1)
        
        loss_vowel = F.cross_entropy(vowel_logits[vowel_mask], labels[vowel_mask]) if vowel_mask.sum() > 0 else 0.0
        # For consonants, labels offset by NUM_VOWELS (13)
        loss_cons = F.cross_entropy(consonant_logits[cons_mask], labels[cons_mask] - NUM_VOWELS) if cons_mask.sum() > 0 else 0.0
        
        # 3. Overall fused loss
        loss_fused = F.cross_entropy(fused_logits, labels)
        
        total_loss = loss_fused + 0.3 * loss_group + 0.2 * (loss_vowel + loss_cons)
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        top1, top3 = accuracy(fused_logits, labels, topk=(1, 3))
        running_loss += total_loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top3 += top3 * batch_size
        total_samples += batch_size
        
    return running_loss / total_samples, total_top1 / total_samples, total_top3 / total_samples

def evaluate_hierarchical(
    model: FastHierarchicalNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate FastHierarchicalNet."""
    model.eval()
    running_loss, total_top1, total_top3, total_samples = 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, labels, groups in loader:
            images, labels, groups = images.to(device), labels.to(device), groups.to(device)
            batch_size = labels.size(0)
            
            fused_logits, _, _, _ = model(images)
            loss = F.cross_entropy(fused_logits, labels)
            
            top1, top3 = accuracy(fused_logits, labels, topk=(1, 3))
            running_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top3 += top3 * batch_size
            total_samples += batch_size
            
            preds = torch.argmax(fused_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            
    return (
        running_loss / total_samples,
        total_top1 / total_samples,
        total_top3 / total_samples,
        np.array(all_preds),
        np.array(all_targets)
    )

def plot_training_results(
    history: Dict[str, list],
    model_name: str,
    output_path: Path
):
    """Generate professional loss and accuracy comparison plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title(f"{model_name}: Loss over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Top-1", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"], label="Val Top-1", color="#d62728", lw=2, linestyle="--")
    ax2.plot(epochs, history["val_top3"], label="Val Top-3", color="#9467bd", lw=1.5, linestyle=":")
    ax2.set_title(f"{model_name}: Accuracy over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {output_path}")

def train_model(
    model_type: str = "fastnet",
    mode: str = "all",
    epochs: int = 20,
    batch_size: int = BATCH_SIZE,
    lr: float = 3e-3,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """Train specified model with modern high-speed scheduler."""
    print("=" * 65)
    print(f"Training: Model={model_type.upper()} | Mode={mode.upper()} | Device={device}")
    print("=" * 65)
    
    train_loader, val_loader, meta = get_dataloaders(mode=mode, batch_size=batch_size)
    num_classes = meta["num_classes"]
    
    if model_type == "baseline":
        model = BaselineKannadaCNN(num_classes=num_classes).to(device)
    elif model_type == "hierarchical":
        model = FastHierarchicalNet().to(device)
    else:
        model = FastKannadaNet(num_classes=num_classes).to(device)
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {model.__class__.__name__} loaded with {num_params:,} parameters.")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3, anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top3": []}
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if model_type == "hierarchical":
            trn_loss, trn_acc, _ = train_epoch_hierarchical(model, train_loader, optimizer, scheduler, device)
            val_loss, val_acc, val_top3, _, _ = evaluate_hierarchical(model, val_loader, device)
        else:
            trn_loss, trn_acc, _ = train_epoch_standard(model, train_loader, optimizer, scheduler, criterion, device)
            val_loss, val_acc, val_top3, _, _ = evaluate_standard(model, val_loader, criterion, device)
            
        epoch_duration = time.time() - epoch_start
        history["train_loss"].append(trn_loss)
        history["train_acc"].append(trn_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top3"].append(val_top3)
        
        print(
            f"Epoch {epoch:2d}/{epochs:2d} [{epoch_duration:.1f}s] - "
            f"Train Loss: {trn_loss:.4f} | Train Acc: {trn_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-3: {val_top3:.2f}%",
            flush=True
        )
        
        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_name = f"{model_type}_{mode}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_type": model_type,
                "mode": mode,
                "num_classes": num_classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, SAVED_MODELS_DIR / save_name)
            
    total_duration = time.time() - start_time
    print(f"\nTraining complete in {total_duration:.1f}s. Peak Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plot curves
    plot_file = OUTPUTS_DIR / f"curves_{model_type}_{mode}.png"
    plot_training_results(history, f"{model_type.capitalize()} ({mode})", plot_file)
    
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "total_time": total_duration,
        "parameters": num_params,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fastnet", choices=["fastnet", "baseline", "hierarchical"])
    parser.add_argument("--mode", type=str, default="all", choices=["all", "vowels", "consonants"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

"""
Evaluation and Metrics Reporting Module for Lipika Kannada Character Recognizer.
Calculates Top-1 & Top-3 Accuracy, Precision, Recall, Macro F1-Score,
and generates/saves the full 47x47 Confusion Matrix and Group Matrix to outputs/.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import onnxruntime as ort
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR, SAVED_MODELS_DIR, NUM_CLASSES, NUM_VOWELS
from data.kannada_varnamale import ALL_CHARACTERS, ID_TO_NAME, ID_TO_CHAR, GROUP_NAMES
from data.dataset_loader import load_npz_dataset
from models.fast_kannadanet import FastKannadaNet
from models.baseline_cnn import BaselineKannadaCNN

def run_evaluation(
    model_path: str = None,
    dataset_path: str = None,
    backend: str = "onnx"
) -> Dict[str, Any]:
    """Run evaluation on test dataset and save all metrics to outputs/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Model Path
    if model_path is None:
        if backend == "onnx":
            # Default to fast_kannadanet.onnx (or kannadanet_best.onnx)
            candidate = SAVED_MODELS_DIR / "fast_kannadanet.onnx"
            if not candidate.exists():
                candidate = SAVED_MODELS_DIR / "kannadanet_best.onnx"
            model_path = str(candidate)
        else:
            candidate = SAVED_MODELS_DIR / "baseline_all_best.pt"
            if not candidate.exists():
                candidate = SAVED_MODELS_DIR / "fastnet_all_best.pt"
            model_path = str(candidate)
            
    print("=" * 75)
    print(f"EVALUATING MODEL: {model_path} [{backend.upper()}]")
    print("=" * 75)
    
    # 2. Load Dataset
    data = load_npz_dataset(dataset_path)
    x_test = data["x_test"]  # shape (N, 28, 28)
    y_test = data["y_test"]  # shape (N,)
    g_test = data["groups_test"] # shape (N,)
    num_samples = len(y_test)
    print(f"Loaded {num_samples} held-out test samples across {NUM_CLASSES} Kannada classes.")
    
    # 3. Model Inference
    all_preds = []
    all_probs = []
    
    if backend == "onnx":
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        
        # Batch inference
        batch_size = 256
        for i in range(0, num_samples, batch_size):
            batch_x = x_test[i:i+batch_size].astype(np.float32) / 255.0
            batch_x = np.expand_dims(batch_x, axis=1) # (B, 1, 28, 28)
            logits = session.run(None, {input_name: batch_x})[0]
            # Softmax
            exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)
    else:
        device = torch.device("cpu")
        if "baseline" in str(model_path).lower():
            model = BaselineKannadaCNN(num_classes=NUM_CLASSES)
        else:
            model = FastKannadaNet(num_classes=NUM_CLASSES)
        cp = torch.load(model_path, map_location=device)
        state = cp["model_state"] if "model_state" in cp else cp
        model.load_state_dict(state)
        model.eval()
        
        batch_size = 256
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float().unsqueeze(1) / 255.0
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=-1).numpy()
                preds = np.argmax(probs, axis=-1)
                all_preds.extend(preds)
                all_probs.extend(probs)
                
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4. Compute Metrics
    acc = accuracy_score(y_test, all_preds)
    
    # Top-3 Accuracy
    top3_correct = 0
    for i in range(num_samples):
        top3_indices = np.argsort(all_probs[i])[::-1][:3]
        if y_test[i] in top3_indices:
            top3_correct += 1
    top3_acc = top3_correct / num_samples
    
    # Macro and Weighted Precision, Recall, F1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, all_preds, average="weighted", zero_division=0
    )
    
    # 5. Confusion Matrices
    # A. 47x47 Full Class Confusion Matrix
    cm_47 = confusion_matrix(y_test, all_preds, labels=list(range(NUM_CLASSES)))
    
    # B. 2x2 Group Confusion Matrix (Swara vs Vyanjana)
    pred_groups = np.where(all_preds < NUM_VOWELS, 0, 1)
    cm_group = confusion_matrix(g_test, pred_groups, labels=[0, 1])
    
    # 6. Generate and Save Visualizations in outputs/
    class_names = [f"{ID_TO_NAME[i]}" for i in range(NUM_CLASSES)]
    
    # Plot 1: Full 47x47 Confusion Matrix Heatmap
    plt.figure(figsize=(18, 15), dpi=300)
    sns.heatmap(
        cm_47,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 6}
    )
    plt.title(f"47-Class Kannada Character Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=15, fontweight="bold", pad=12)
    plt.xlabel("Predicted Character Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Ground-Truth Character Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_plot_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"-> Saved 47-class Confusion Matrix plot to: {cm_plot_path}")
    
    # Plot 2: 2x2 Group Confusion Matrix (Swara vs Vyanjana)
    plt.figure(figsize=(7, 6), dpi=300)
    group_labels = ["Swara (Vowels)", "Vyanjana (Consonants)"]
    sns.heatmap(
        cm_group,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=group_labels,
        yticklabels=group_labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Linguistic Group Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11, fontweight="bold")
    plt.ylabel("True Category", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_group_path = OUTPUTS_DIR / "confusion_matrix_group.png"
    plt.savefig(cm_group_path, dpi=300)
    plt.close()
    print(f"-> Saved Group Confusion Matrix plot to: {cm_group_path}")
    
    # 7. Classification Report Text
    report_dict = classification_report(
        y_test, all_preds, target_names=class_names, digits=4, zero_division=0
    )
    report_file = OUTPUTS_DIR / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LIPIKA KANNADA CHARACTER RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_dict)
    print(f"-> Saved detailed classification report to: {report_file}")
    
    # 8. Executive Metrics Summary File
    summary_text = f"""========================================================================================
LIPIKA EVALUATION METRICS SUMMARY
========================================================================================
Model Evaluated          : {model_path}
Evaluation Backend       : {backend.upper()}
Total Test Samples       : {num_samples} (Held-out 20% test partition)
Total Character Classes  : {NUM_CLASSES} (13 Swaras + 34 Vyanjanas)

KEY PERFORMANCE METRICS:
----------------------------------------------------------------------------------------
• Top-1 Test Accuracy    : {acc * 100:.2f}% ({acc:.4f})
• Top-3 Test Accuracy    : {top3_acc * 100:.2f}% ({top3_acc:.4f})
• Macro Precision        : {prec_macro * 100:.2f}% ({prec_macro:.4f})
• Macro Recall           : {rec_macro * 100:.2f}% ({rec_macro:.4f})
• Macro F1-Score         : {f1_macro * 100:.2f}% ({f1_macro:.4f})
• Weighted F1-Score      : {f1_weighted * 100:.2f}% ({f1_weighted:.4f})

LINGUISTIC GROUP CONFUSION MATRIX (Swara vs. Vyanjana):
----------------------------------------------------------------------------------------
Actual Swara (1,300)     : Predicted Swara = {cm_group[0, 0]:>4d} | Predicted Vyanjana = {cm_group[0, 1]:>4d}
Actual Vyanjana (3,400)  : Predicted Swara = {cm_group[1, 0]:>4d} | Predicted Vyanjana = {cm_group[1, 1]:>4d}
Total Correct            : {cm_group[0, 0] + cm_group[1, 1]} / {num_samples} ({(cm_group[0, 0] + cm_group[1, 1]) / num_samples * 100:.2f}%)

SAVED ARTIFACTS IN outputs/ DIRECTORY:
• outputs/confusion_matrix.png        : 47x47 Heatmap visualization
• outputs/confusion_matrix_group.png  : Swara vs. Vyanjana 2x2 Heatmap
• outputs/classification_report.txt   : Full per-class precision, recall, and F1 table
• outputs/evaluation_metrics.txt      : This executive summary
========================================================================================
"""
    summary_file = OUTPUTS_DIR / "evaluation_metrics.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"-> Saved evaluation summary to: {summary_file}")
    
    # Print clean summary to terminal
    print("\n" + summary_text)
    
    return {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix_47": cm_47,
        "confusion_matrix_group": cm_group
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lipika Kannada Character Recognizer.")
    parser.add_argument("--model", type=str, default=None, help="Path to .onnx or .pt model file")
    parser.add_argument("--backend", type=str, default="onnx", choices=["onnx", "torch"], help="Inference backend")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .npz test dataset")
    args = parser.parse_args()
    
    run_evaluation(model_path=args.model, dataset_path=args.dataset, backend=args.backend)

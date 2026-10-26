"""
ONNX Export Utility for Lipika Character Recognizers.
Exports trained models (Baseline CNN or Fast-KannadaNet) to ONNX Runtime.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import NUM_CLASSES, SAVED_MODELS_DIR
from models.fast_kannadanet import FastKannadaNet
from models.baseline_cnn import BaselineKannadaCNN

def export_model_to_onnx(
    model_type: str = "baseline",
    weights_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """Export trained model to ONNX."""
    if output_path is None:
        output_path = str(SAVED_MODELS_DIR / "kannadanet_best.onnx")
        
    if model_type == "baseline":
        model = BaselineKannadaCNN(num_classes=NUM_CLASSES)
        default_weights = str(SAVED_MODELS_DIR / "baseline_all_best.pt")
    else:
        model = FastKannadaNet(num_classes=NUM_CLASSES)
        default_weights = str(SAVED_MODELS_DIR / "fastnet_all_best.pt")
        
    weights = weights_path or default_weights
    if os.path.exists(weights):
        cp = torch.load(weights, map_location="cpu")
        state = cp["model_state"] if "model_state" in cp else cp
        model.load_state_dict(state)
        print(f"Loaded {model_type} weights from {weights}")
    else:
        print(f"Warning: Weights not found at {weights}, using initialized weights.")
        
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    
    # Also write to fast_kannadanet.onnx so web_app automatically picks it up
    web_onnx_path = str(SAVED_MODELS_DIR / "fast_kannadanet.onnx")
    import shutil
    shutil.copyfile(output_path, web_onnx_path)
    
    # Validate with onnxruntime
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    print(f"ONNX export successful: {output_path} (Output shape: {ort_outs[0].shape})")
    
    return output_path

if __name__ == "__main__":
    export_model_to_onnx(model_type="baseline")

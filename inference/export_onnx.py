"""
ONNX & TorchScript Export Utility for Lipika Character Recognizers.
Enables sub-millisecond, low-latency CPU and edge deployment.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
from models.hierarchical_net import FastHierarchicalNet

def export_fastnet_to_onnx(
    weights_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """Export FastKannadaNet to ONNX using legacy/standard engine."""
    if output_path is None:
        output_path = str(SAVED_MODELS_DIR / "fast_kannadanet.onnx")
        
    model = FastKannadaNet(num_classes=NUM_CLASSES)
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Using initialized model weights for export.")
        
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    
    # Use dynamo=False for reliable classic ONNX export
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
    
    # Check ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported Fast-KannadaNet to ONNX: {output_path}")
    
    # Test with onnxruntime
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    print(f"ONNX test inference successful! Output shape: {ort_outs[0].shape}")
    
    return output_path

if __name__ == "__main__":
    best_weights = str(SAVED_MODELS_DIR / "fastnet_all_best.pt")
    export_fastnet_to_onnx(weights_path=best_weights if os.path.exists(best_weights) else None)

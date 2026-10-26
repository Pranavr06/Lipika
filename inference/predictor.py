"""
High-Speed Kannada Handwritten Character Predictor.
Supports PyTorch (.pt) and ONNX Runtime (.onnx) backends with
automatic stroke thresholding, bounding-box centering, and Top-K ranking.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import onnxruntime as ort

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import NUM_CLASSES, SAVED_MODELS_DIR, DEVICE
from data.kannada_varnamale import ALL_CHARACTERS, ID_TO_GROUP, GROUP_NAMES, get_character_info
from models.fast_kannadanet import FastKannadaNet
from models.baseline_cnn import BaselineKannadaCNN

class KannadaPredictor:
    """Predictor engine for offline Kannada handwritten character recognition."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "onnx", # "onnx" or "torch"
        device: torch.device = DEVICE
    ):
        self.backend = backend
        self.device = device
        
        if model_path is None:
            if backend == "onnx":
                model_path = str(SAVED_MODELS_DIR / "fast_kannadanet.onnx")
            else:
                model_path = str(SAVED_MODELS_DIR / "fastnet_all_best.pt")
                
        self.model_path = model_path
        
        if backend == "onnx":
            if not os.path.exists(model_path):
                from inference.export_onnx import export_fastnet_to_onnx
                export_fastnet_to_onnx(output_path=model_path)
            self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.ort_session.get_inputs()[0].name
            print(f"KannadaPredictor initialized with ONNX Runtime: {model_path}")
        else:
            self.model = FastKannadaNet(num_classes=NUM_CLASSES)
            if os.path.exists(model_path):
                cp = torch.load(model_path, map_location=device)
                state = cp["model_state"] if "model_state" in cp else cp
                self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()
            print(f"KannadaPredictor initialized with PyTorch: {model_path} on {device}")

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess input image to normalized 28x28 foreground strokes:
        1. Convert to grayscale.
        2. Detect background (dark or light) and ensure bright strokes on dark background.
        3. Otsu binarization to clean paper texture / noise.
        4. Bounding box cropping around strokes.
        5. Aspect-ratio preserving padding onto 28x28 canvas.
        6. Normalized float32 array in [0.0, 1.0].
        """
        # Load / convert to uint8 grayscale
        if isinstance(image_input, str):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("L"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                img = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            else:
                img = image_input.astype(np.uint8)
        else:
            raise TypeError("Unsupported image input type!")

        # Invert if light background (e.g. white paper with dark ink)
        # Check mean intensity along borders
        border_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        if np.mean(border_pixels) > 127:
            img = cv2.bitwise_not(img)

        # Otsu thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Bounding box crop
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            pad = max(2, int(0.1 * max(bw, bh)))
            y1 = max(0, by - pad)
            y2 = min(img.shape[0], by + bh + pad)
            x1 = max(0, bx - pad)
            x2 = min(img.shape[1], bx + bw + pad)
            cropped = img[y1:y2, x1:x2]
        else:
            cropped = img

        ch, cw = cropped.shape
        if ch == 0 or cw == 0:
            cropped = img
            ch, cw = cropped.shape

        # Scale into 22x22 box inside 28x28 canvas
        scale = 22.0 / max(ch, cw)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.float32)
        ox = (28 - nw) // 2
        oy = (28 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized / 255.0

        return canvas

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Perform inference and return Top-K predicted Kannada characters with confidence scores.
        """
        t0 = time.perf_counter()
        processed_canvas = self.preprocess_image(image_input)
        tensor_input = processed_canvas.reshape(1, 1, 28, 28).astype(np.float32)

        if self.backend == "onnx":
            outputs = self.ort_session.run(None, {self.input_name: tensor_input})
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            with torch.no_grad():
                inp = torch.from_numpy(tensor_input).to(self.device)
                logits = self.model(inp)[0]
                probs = F.softmax(logits, dim=-1).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            info = get_character_info(int(idx))
            info["confidence"] = float(probs[idx])
            info["confidence_pct"] = f"{probs[idx] * 100.0:.2f}%"
            top_predictions.append(info)

        return {
            "top_predictions": top_predictions,
            "best_character": top_predictions[0]["char"],
            "best_phonetic": top_predictions[0]["phonetic"],
            "best_confidence": top_predictions[0]["confidence"],
            "latency_ms": latency_ms,
            "processed_image": processed_canvas,
        }

if __name__ == "__main__":
    predictor = KannadaPredictor(backend="onnx")
    # Quick test on random dummy canvas
    dummy = np.zeros((28, 28), dtype=np.uint8)
    dummy[8:20, 8:20] = 255
    res = predictor.predict(dummy)
    print("Test Prediction:")
    print("Best char:", res["best_character"], f"({res['best_phonetic']})", "Confidence:", res["top_predictions"][0]["confidence_pct"])
    print(f"Latency: {res['latency_ms']:.2f} ms")

"""
Document & Word Segmentation Pipeline for Offline Kannada HCR.
Segments handwritten Kannada lines, words, and characters into individual bounding boxes
and transcribes them into Unicode strings using Fast-KannadaNet.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from inference.predictor import KannadaPredictor

class KannadaOCR:
    """End-to-End Offline Kannada Handwritten Character & Word Recognizer."""
    def __init__(self, predictor: Optional[KannadaPredictor] = None):
        self.predictor = predictor or KannadaPredictor(backend="onnx")

    def segment_characters(
        self,
        image: np.ndarray,
        min_area: int = 40
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Segment handwritten characters using connected component analysis.
        Returns:
            crops: list of 28x28 normalized character images.
            boxes: list of (x, y, w, h) bounding boxes sorted left-to-right.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Invert if light background
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if np.mean(border) > 127:
            binary = cv2.bitwise_not(gray)
        else:
            binary = gray.copy()

        # Otsu thresholding
        _, thresh = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological horizontal closing to keep complex Kannada conjunct strokes together
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find connected components / contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area and w > 4 and h > 4:
                boxes.append((x, y, w, h))

        # Sort left-to-right (primary) with line clustering tolerance (secondary)
        boxes = sorted(boxes, key=lambda b: (b[1] // 30, b[0]))

        crops = []
        valid_boxes = []
        for (x, y, w, h) in boxes:
            char_crop = binary[max(0, y - 2):min(binary.shape[0], y + h + 2), max(0, x - 2):min(binary.shape[1], x + w + 2)]
            if char_crop.size == 0:
                continue
            preprocessed = self.predictor.preprocess_image(char_crop)
            crops.append(preprocessed)
            valid_boxes.append((x, y, w, h))

        return crops, valid_boxes

    def recognize_document(self, image_input: Any) -> Dict[str, Any]:
        """Recognize full image and transcribe all Kannada characters."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            img = image_input

        crops, boxes = self.segment_characters(img)
        if not crops:
            return {"text": "", "characters": [], "count": 0}

        results = []
        text_chars = []
        for crop, box in zip(crops, boxes):
            pred = self.predictor.predict(crop, top_k=3)
            best_char = pred["best_character"]
            text_chars.append(best_char)
            results.append({
                "box": box,
                "character": best_char,
                "phonetic": pred["best_phonetic"],
                "confidence": pred["best_confidence"],
                "top_predictions": pred["top_predictions"],
            })

        return {
            "text": "".join(text_chars),
            "characters": results,
            "count": len(results),
        }

if __name__ == "__main__":
    ocr = KannadaOCR()
    print("KannadaOCR pipeline initialized successfully!")

"""
Unit and Integration Test Suite for Lipika.
Tests character definitions, model architectures, dataset pipelines, and inference engines.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import unittest
import numpy as np
import torch

from data.kannada_varnamale import (
    ALL_CHARACTERS,
    SWARAS,
    VYANJANAS,
    NUM_CLASSES,
    NUM_VOWELS,
    NUM_CONSONANTS,
    ID_TO_CHAR,
    ID_TO_GROUP,
    get_character_info
)
from models.baseline_cnn import BaselineKannadaCNN
from models.fast_kannadanet import FastKannadaNet
from models.hierarchical_net import FastHierarchicalNet
from inference.predictor import KannadaPredictor
from inference.segmenter import KannadaOCR

class TestLipikaPipeline(unittest.TestCase):
    def test_character_varnamale(self):
        """Test linguistic varnamale integrity."""
        self.assertEqual(len(SWARAS), 13, "Should have exactly 13 vowels.")
        self.assertEqual(len(VYANJANAS), 34, "Should have exactly 34 consonants.")
        self.assertEqual(NUM_CLASSES, 47, "Total classes must equal 47.")
        self.assertEqual(len(ALL_CHARACTERS), 47)
        self.assertEqual(len(set(ID_TO_CHAR.values())), 47, "All glyphs must be unique.")
        
        # Test first vowel is 'a' (ಅ) and first consonant is 'ka' (ಕ)
        self.assertEqual(ID_TO_CHAR[0], "ಅ")
        self.assertEqual(ID_TO_CHAR[13], "ಕ")
        self.assertEqual(ID_TO_GROUP[0], 0)  # Vowel group
        self.assertEqual(ID_TO_GROUP[13], 1) # Consonant group

    def test_baseline_cnn_forward(self):
        """Test Baseline CNN architecture."""
        model = BaselineKannadaCNN(num_classes=47)
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (2, 47))
        self.assertGreater(model.count_parameters(), 700_000)

    def test_fast_kannadanet_forward(self):
        """Test Fast-KannadaNet architecture."""
        model = FastKannadaNet(num_classes=47)
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        features = model.extract_features(x)
        
        self.assertEqual(out.shape, (2, 47))
        self.assertEqual(features.shape, (2, 128))
        self.assertLess(model.count_parameters(), 200_000, "FastNet must be < 200K parameters.")

    def test_hierarchical_net_forward(self):
        """Test FastHierarchicalNet architecture."""
        model = FastHierarchicalNet()
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        fused, group, vowels, consonants = model(x)
        
        self.assertEqual(fused.shape, (2, 47))
        self.assertEqual(group.shape, (2, 2))
        self.assertEqual(vowels.shape, (2, 13))
        self.assertEqual(consonants.shape, (2, 34))

    def test_onnx_predictor(self):
        """Test ONNX Runtime predictor on synthetic character canvas."""
        predictor = KannadaPredictor(backend="onnx")
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[5:23, 5:23] = 255
        
        res = predictor.predict(canvas, top_k=3)
        self.assertIn("best_character", res)
        self.assertIn("latency_ms", res)
        self.assertEqual(len(res["top_predictions"]), 3)
        self.assertLess(res["latency_ms"], 50.0, "Latency should be very low.")

    def test_ocr_segmenter(self):
        """Test document segmenter."""
        ocr = KannadaOCR()
        test_img = np.zeros((60, 120), dtype=np.uint8)
        # Draw two simulated character blocks
        test_img[10:50, 10:40] = 255
        test_img[10:50, 60:90] = 255
        
        doc_res = ocr.recognize_document(test_img)
        self.assertEqual(doc_res["count"], 2)
        self.assertEqual(len(doc_res["characters"]), 2)

if __name__ == "__main__":
    unittest.main()

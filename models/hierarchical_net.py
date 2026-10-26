"""
Fast Hierarchical Routing Network for Kannada Character Recognition.

Implements the key architectural breakthrough identified in the 2019 IEEE Paper:
Separating Vowels (Swaras - 13 classes) and Consonants (Vyanjanas - 34 classes)
eliminates cross-family feature interference, boosting recognition accuracy from ~78% to >93%.

FastHierarchicalNet achieves this in a SINGLE, unified, end-to-end differentiable network
with joint routing:
1. Shared lightweight Fast-KannadaNet backbone (Depthwise Separable + SE).
2. Group Router Head: Predicts Vowel vs. Consonant (2 classes).
3. Specialized Vowel Sub-Head: Classifies among 13 Swaras.
4. Specialized Consonant Sub-Head: Classifies among 34 Vyanjanas.
5. Marginal Probability Fusion: P(Char_i) = P(Group_k) * P(Char_i | Group_k).
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fast_kannadanet import FastKannadaNet
from config import NUM_CLASSES, NUM_VOWELS, NUM_CONSONANTS

class FastHierarchicalNet(nn.Module):
    """
    Unified Hierarchical Character Recognizer:
    Backbone -> [Group Router Head (2)]
             -> [Vowel Expert Head (13)]
             -> [Consonant Expert Head (34)]
    """
    def __init__(self, dropout_rate: float = 0.20):
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.num_vowels = NUM_VOWELS
        self.num_consonants = NUM_CONSONANTS
        
        # Shared high-speed feature extractor
        self.backbone = FastKannadaNet(num_classes=NUM_CLASSES, dropout_rate=dropout_rate)
        feature_dim = 128
        
        # Dual Heads
        self.group_classifier = nn.Linear(feature_dim, 2)              # 0: Vowel, 1: Consonant
        self.vowel_classifier = nn.Linear(feature_dim, NUM_VOWELS)      # 13 Swaras
        self.consonant_classifier = nn.Linear(feature_dim, NUM_CONSONANTS) # 34 Vyanjanas
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            fused_logits: (B, 47) combined probabilities over all classes.
            group_logits: (B, 2) Vowel vs Consonant logits.
            vowel_logits: (B, 13) Vowel logits.
            consonant_logits: (B, 34) Consonant logits.
        """
        features = self.backbone.extract_features(x)
        features = self.dropout(features)
        
        group_logits = self.group_classifier(features)
        vowel_logits = self.vowel_classifier(features)
        consonant_logits = self.consonant_classifier(features)
        
        # Marginal probability fusion
        group_probs = F.softmax(group_logits, dim=-1) # (B, 2)
        vowel_probs = F.softmax(vowel_logits, dim=-1) # (B, 13)
        cons_probs = F.softmax(consonant_logits, dim=-1) # (B, 34)
        
        # P(char_vowel) = P(is_vowel) * P(vowel_i)
        p_vowels = group_probs[:, 0:1] * vowel_probs
        # P(char_cons) = P(is_cons) * P(cons_j)
        p_consonants = group_probs[:, 1:2] * cons_probs
        
        # Concatenate into full 47-class probability distribution
        fused_probs = torch.cat([p_vowels, p_consonants], dim=-1)
        # Convert to pseudo-logits for numerical stability in cross-entropy loss
        fused_logits = torch.log(torch.clamp(fused_probs, min=1e-7))
        
        return fused_logits, group_logits, vowel_logits, consonant_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

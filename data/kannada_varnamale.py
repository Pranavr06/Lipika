"""
Kannada Varnamale (ಕನ್ನಡ ವರ್ಣಮಾಲೆ) Character Definitions and Mappings.

Consists of 47 fundamental characters according to the research paper:
- 13 Swaras (Vowels)
- 34 Vyanjanas (Consonants: 25 Vargiya + 9 Avargiya)
"""

from typing import Dict, List, Tuple

# 13 Swaras (Vowels)
SWARAS: List[Tuple[str, str, str]] = [
    ("\u0C85", "a", "ಅ"),
    ("\u0C86", "aa", "ಆ"),
    ("\u0C87", "i", "ಇ"),
    ("\u0C88", "ee", "ಈ"),
    ("\u0C89", "u", "ಉ"),
    ("\u0C8A", "oo", "ಊ"),
    ("\u0C8B", "ru", "ಋ"),
    ("\u0C8E", "e", "ಎ"),
    ("\u0C8F", "ae", "ಏ"),
    ("\u0C90", "ai", "ಐ"),
    ("\u0C92", "o", "ಒ"),
    ("\u0C93", "oa", "ಓ"),
    ("\u0C94", "au", "ಔ"),
]

# 34 Vyanjanas (Consonants)
# 25 Vargiya (Structured) Consonants + 9 Avargiya (Unstructured) Consonants
VYANJANAS: List[Tuple[str, str, str]] = [
    # Ka-varga (Velar)
    ("\u0C95", "ka", "ಕ"),
    ("\u0C96", "kha", "ಖ"),
    ("\u0C97", "ga", "ಗ"),
    ("\u0C98", "gha", "ಘ"),
    ("\u0C99", "nga", "ಙ"),
    # Cha-varga (Palatal)
    ("\u0C9A", "cha", "ಚ"),
    ("\u0C9B", "chha", "ಛ"),
    ("\u0C9C", "ja", "ಜ"),
    ("\u0C9D", "jha", "ಝ"),
    ("\u0C9E", "nya", "ಞ"),
    # Ta-varga (Retroflex)
    ("\u0C9F", "tta", "ಟ"),
    ("\u0CA0", "ttha", "ಠ"),
    ("\u0CA1", "dda", "ಡ"),
    ("\u0CA2", "ddha", "ಢ"),
    ("\u0CA3", "nna", "ಣ"),
    # Tha-varga (Dental)
    ("\u0CA4", "ta", "ತ"),
    ("\u0CA5", "tha", "ಥ"),
    ("\u0CA6", "da", "ದ"),
    ("\u0CA7", "dha", "ಧ"),
    ("\u0CA8", "na", "ನ"),
    # Pa-varga (Labial)
    ("\u0CAA", "pa", "ಪ"),
    ("\u0CAB", "pha", "ಫ"),
    ("\u0CAC", "ba", "ಬ"),
    ("\u0CAD", "bha", "ಭ"),
    ("\u0CAE", "ma", "ಮ"),
    # Avargiya Vyanjanas (Unstructured)
    ("\u0CAF", "ya", "ಯ"),
    ("\u0CB0", "ra", "ರ"),
    ("\u0CB2", "la", "ಲ"),
    ("\u0CB5", "va", "ವ"),
    ("\u0CB6", "sha", "ಶ"),
    ("\u0CB7", "ssa", "ಷ"),
    ("\u0CB8", "sa", "ಸ"),
    ("\u0CB9", "ha", "ಹ"),
    ("\u0CB3", "lla", "ಳ"),
]

# Total 47 Classes
ALL_CHARACTERS = SWARAS + VYANJANAS
NUM_VOWELS = len(SWARAS)         # 13
NUM_CONSONANTS = len(VYANJANAS)  # 34
NUM_CLASSES = len(ALL_CHARACTERS) # 47

# Lookup Mappings
ID_TO_CHAR: Dict[int, str] = {i: item[0] for i, item in enumerate(ALL_CHARACTERS)}
ID_TO_NAME: Dict[int, str] = {i: item[1] for i, item in enumerate(ALL_CHARACTERS)}
CHAR_TO_ID: Dict[str, int] = {item[0]: i for i, item in enumerate(ALL_CHARACTERS)}
NAME_TO_ID: Dict[str, int] = {item[1]: i for i, item in enumerate(ALL_CHARACTERS)}

# Group IDs: 0 = Vowel (Swara), 1 = Consonant (Vyanjana)
ID_TO_GROUP: Dict[int, int] = {i: (0 if i < NUM_VOWELS else 1) for i in range(NUM_CLASSES)}
GROUP_NAMES = ["Swara (Vowel)", "Vyanjana (Consonant)"]

def get_character_info(class_id: int) -> Dict[str, str]:
    """Retrieve character information dictionary by class ID."""
    if not (0 <= class_id < NUM_CLASSES):
        raise ValueError(f"Invalid class ID: {class_id}. Must be in [0, {NUM_CLASSES - 1}].")
    char_glyph, phonetic, display = ALL_CHARACTERS[class_id]
    group_id = ID_TO_GROUP[class_id]
    return {
        "class_id": class_id,
        "char": char_glyph,
        "phonetic": phonetic,
        "display": display,
        "group_id": group_id,
        "group_name": GROUP_NAMES[group_id],
        "unicode": f"U+{ord(char_glyph):04X}",
    }

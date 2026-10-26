"""
Realistic Handwritten Kannada Character Dataset Generator.
Synthesizes handwritten Kannada characters (47 classes: 13 Swaras + 34 Vyanjanas)
with human handwriting simulations (elastic spline deformation, stroke thickness jitter,
affine transformations, perspective variations, and morphology).
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.ndimage import map_coordinates, gaussian_filter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.kannada_varnamale import ALL_CHARACTERS, NUM_CLASSES, ID_TO_GROUP
from config import FONTS_DIR, PROCESSED_DATA_DIR, SEED

def get_available_fonts() -> List[str]:
    """Collect all available Kannada fonts."""
    font_candidates = [
        str(FONTS_DIR / "NotoSansKannada.ttf"),
        str(FONTS_DIR / "AnekKannada.ttf"),
        str(FONTS_DIR / "Hubballi.ttf"),
        str(FONTS_DIR / "BalooTamma2.ttf"),
        str(FONTS_DIR / "TiroKannada.ttf"),
        "C:/Windows/Fonts/Nirmala.ttc",
    ]
    available = [f for f in font_candidates if os.path.exists(f)]
    if not available:
        raise FileNotFoundError("No Kannada fonts found in fonts/ or C:/Windows/Fonts/.")
    return available

def elastic_transform(image: np.ndarray, alpha: float = 18.0, sigma: float = 3.0) -> np.ndarray:
    """
    Simulate natural handwriting tremors and hand pressure variations
    using elastic displacement fields (Simard et al.).
    """
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(image, indices, order=1, mode="reflect")
    return distorted.reshape(shape)

def render_character_glyph(
    char: str,
    font_path: str,
    canvas_size: int = 64,
    font_size: int = 42
) -> np.ndarray:
    """Render a single Kannada character glyph centered on a high-res grayscale canvas."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.truetype(font_path, font_size, index=0)
        
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Get bounding box to center the glyph
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    x_pos = (canvas_size - w) // 2 - bbox[0]
    y_pos = (canvas_size - h) // 2 - bbox[1]
    
    draw.text((x_pos, y_pos), char, fill=255, font=font)
    return np.array(img, dtype=np.uint8)

def apply_handwriting_distortions(glyph: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Apply realistic handwriting augmentations:
    - Random rotation (-14 to +14 deg)
    - Random shear (-10 to +10 deg)
    - Stroke dilation (thick ballpoint/gel pen) or erosion (fine nib)
    - Elastic muscular jitter
    - Centering and aspect-ratio preserved downsampling to 28x28
    """
    h, w = glyph.shape
    
    # 1. Stroke thickness jitter
    kernel_size = random.choice([0, 2, 3])
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if random.random() < 0.65:
            glyph = cv2.dilate(glyph, kernel, iterations=1)
        else:
            glyph = cv2.erode(glyph, kernel, iterations=1)
            
    # 2. Affine transformations (rotation, translation, shear)
    angle = random.uniform(-14.0, 14.0)
    shear_x = random.uniform(-0.15, 0.15)
    shear_y = random.uniform(-0.10, 0.10)
    scale = random.uniform(0.85, 1.15)
    
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0, 1] += shear_x
    rot_mat[1, 0] += shear_y
    rot_mat[0, 2] += random.uniform(-3, 3)
    rot_mat[1, 2] += random.uniform(-3, 3)
    
    transformed = cv2.warpAffine(glyph, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 3. Elastic handwriting deformation (applied with 60% probability)
    if random.random() < 0.60:
        transformed = elastic_transform(transformed, alpha=random.uniform(8.0, 16.0), sigma=random.uniform(2.5, 3.5))
        
    # 4. Find bounding box of foreground strokes to center accurately
    coords = cv2.findNonZero((transformed > 30).astype(np.uint8))
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        # Pad slightly
        pad = max(2, int(0.1 * max(bw, bh)))
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)
        cropped = transformed[y1:y2, x1:x2]
    else:
        cropped = transformed
        
    # 5. Aspect-ratio preserving resize onto target 28x28 canvas
    ch, cw = cropped.shape
    if ch == 0 or cw == 0:
        cropped = glyph
        ch, cw = cropped.shape
        
    max_dim = max(ch, cw)
    scale_factor = (target_size - 6) / float(max_dim)
    new_w = max(1, int(cw * scale_factor))
    new_h = max(1, int(ch * scale_factor))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    final_canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    ox = (target_size - new_w) // 2
    oy = (target_size - new_h) // 2
    final_canvas[oy:oy+new_h, ox:ox+new_w] = resized
    
    # Squeezed pixel normalization
    return final_canvas

def generate_kannada_dataset(
    samples_per_class: int = 500,
    train_ratio: float = 0.8,
    target_size: int = 28,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dataset matching the IEEE paper specification:
    - 47 characters (13 Swaras + 34 Vyanjanas)
    - 500 samples per class = 23,500 total images
    - 80:20 Train:Test split (18,800 train, 4,700 test)
    """
    np.random.seed(SEED)
    random.seed(SEED)
    
    fonts = get_available_fonts()
    print(f"Generating Kannada HCR dataset using {len(fonts)} authentic fonts...")
    
    x_train_list, y_train_list, g_train_list = [], [], []
    x_test_list, y_test_list, g_test_list = [], [], []
    
    n_train = int(samples_per_class * train_ratio)
    n_test = samples_per_class - n_train
    
    for class_id in tqdm(range(NUM_CLASSES), desc="Synthesizing Kannada Classes"):
        char_glyph, phonetic, _ = ALL_CHARACTERS[class_id]
        group_id = ID_TO_GROUP[class_id] # 0 = vowel, 1 = consonant
        
        # Pre-render base glyphs from all fonts
        base_glyphs = [render_character_glyph(char_glyph, f, canvas_size=64, font_size=40) for f in fonts]
        
        # Train samples
        for _ in range(n_train):
            base = random.choice(base_glyphs)
            sample = apply_handwriting_distortions(base, target_size=target_size)
            x_train_list.append(sample)
            y_train_list.append(class_id)
            g_train_list.append(group_id)
            
        # Test samples
        for _ in range(n_test):
            base = random.choice(base_glyphs)
            sample = apply_handwriting_distortions(base, target_size=target_size)
            x_test_list.append(sample)
            y_test_list.append(class_id)
            g_test_list.append(group_id)
            
    x_train = np.array(x_train_list, dtype=np.uint8)
    y_train = np.array(y_train_list, dtype=np.int64)
    g_train = np.array(g_train_list, dtype=np.int64)
    
    x_test = np.array(x_test_list, dtype=np.uint8)
    y_test = np.array(y_test_list, dtype=np.int64)
    g_test = np.array(g_test_list, dtype=np.int64)
    
    # Shuffle train set
    train_perm = np.random.permutation(len(x_train))
    x_train = x_train[train_perm]
    y_train = y_train[train_perm]
    g_train = g_train[train_perm]
    
    # Shuffle test set
    test_perm = np.random.permutation(len(x_test))
    x_test = x_test[test_perm]
    y_test = y_test[test_perm]
    g_test = g_test[test_perm]
    
    if save_path is None:
        save_path = str(PROCESSED_DATA_DIR / f"kannada_dataset_{target_size}x{target_size}.npz")
        
    print(f"Saving compiled dataset to {save_path}...")
    np.savez_compressed(
        save_path,
        x_train=x_train,
        y_train=y_train,
        groups_train=g_train,
        x_test=x_test,
        y_test=y_test,
        groups_test=g_test,
        classes=NUM_CLASSES
    )
    print(f"Dataset generated successfully! Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, g_train, x_test, y_test, g_test

if __name__ == "__main__":
    generate_kannada_dataset()

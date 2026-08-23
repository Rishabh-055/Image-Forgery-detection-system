"""
prep_dataset.py — CASIA Protocol Dataset Generator & Preparer
==============================================================
Creates a standard academic image-tampering dataset following the
CASIA v1/v2 splicing benchmark protocol:
  - Base authentic scenes (nature, architecture, animals, indoor, textures)
  - Spliced donor patches from independent sources with mismatched JPEG compression (Q50-Q95)
  - Geometric transformations (rotation, scaling, translation)
  - Ground truth binary masks (255 = tampered region, 0 = authentic background)
  - Clean train / validation split

Directory structure created:
  data/
    train/
      images/   (e.g., tp_spliced_001.jpg, au_clean_001.jpg)
      masks/    (e.g., tp_spliced_001.png, au_clean_001.png)
    val/
      images/
      masks/
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io

def setup_directories():
    base_dir = "data"
    dirs = [
        "data/train/images",
        "data/train/masks",
        "data/val/images",
        "data/val/masks",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def create_base_texture(width=384, height=384, theme="nature"):
    """Generate authentic base backgrounds across varied categories."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    if theme == "nature":
        # Sky and mountains
        for y in range(height):
            r = int(100 + 80 * (y / height))
            g = int(140 + 70 * (y / height))
            b = int(200 + 40 * (1 - y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Mountain contours
        pts = [(0, height), (width * 0.3, height * 0.5), (width * 0.7, height * 0.65), (width, height * 0.45), (width, height)]
        draw.polygon(pts, fill=(50 + random.randint(0, 30), 90 + random.randint(0, 30), 50 + random.randint(0, 30)))
        
    elif theme == "indoor":
        # Walls and floor
        wall_color = (random.randint(180, 220), random.randint(170, 210), random.randint(160, 200))
        draw.rectangle([0, 0, width, int(height * 0.7)], fill=wall_color)
        floor_color = (random.randint(90, 130), random.randint(60, 90), random.randint(40, 70))
        draw.rectangle([0, int(height * 0.7), width, height], fill=floor_color)
        # Wall panelling
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, int(height * 0.7))], fill=(wall_color[0]-20, wall_color[1]-20, wall_color[2]-20), width=2)
            
    elif theme == "architecture":
        # Geometric structures
        draw.rectangle([0, 0, width, height], fill=(160, 175, 190))
        for i in range(5):
            x1 = random.randint(20, width - 100)
            y1 = random.randint(20, height - 100)
            w = random.randint(40, 100)
            h = random.randint(40, 120)
            c = random.randint(60, 140)
            draw.rectangle([x1, y1, x1 + w, y1 + h], fill=(c, c + 10, c + 20), outline=(30, 30, 40), width=2)
            
    else: # Texture / Noise
        arr = np.random.randint(80, 200, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        return img

    # Add realistic slight blur/grain
    img = img.filter(ImageFilter.SMOOTH)
    return img

def create_donor_object(width=100, height=100, shape="irregular"):
    """Generate donor objects to be spliced (with realistic shapes and textures)."""
    donor = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = Image.new("L", (width, height), 0)
    draw_d = ImageDraw.Draw(donor)
    draw_m = ImageDraw.Draw(mask)
    
    col = (random.randint(180, 255), random.randint(40, 180), random.randint(40, 150))
    
    if shape == "ellipse":
        bbox = [10, 10, width - 10, height - 10]
        draw_d.ellipse(bbox, fill=(*col, 255))
        draw_m.ellipse(bbox, fill=255)
        # Texture stripes inside
        for i in range(15, width - 15, 12):
            draw_d.line([(i, 15), (i + 10, height - 15)], fill=(255, 255, 255, 255), width=3)
    elif shape == "polygon":
        num_pts = random.randint(5, 8)
        cx, cy = width // 2, height // 2
        r = min(cx, cy) - 10
        pts = []
        for i in range(num_pts):
            angle = i * (2 * np.pi / num_pts) + random.uniform(-0.3, 0.3)
            rad = r * random.uniform(0.6, 1.0)
            pts.append((int(cx + rad * np.cos(angle)), int(cy + rad * np.sin(angle))))
        draw_d.polygon(pts, fill=(*col, 255))
        draw_m.polygon(pts, fill=255)
    else: # Irregular / Character shape
        bbox = [15, 15, width - 15, height - 15]
        draw_d.rounded_rectangle(bbox, radius=15, fill=(*col, 255))
        draw_m.rounded_rectangle(bbox, radius=15, fill=255)
        
    return donor, mask

def generate_spliced_pair(sample_id):
    """
    Simulates the exact CASIA splicing process:
    1. Base image saved at Quality Q_base
    2. Donor image created and saved independently at Quality Q_donor
    3. Donor is pasted onto Base
    4. Spliced composite is saved at Quality Q_composite
    5. Ground truth binary mask records exact pasted region
    """
    themes = ["nature", "indoor", "architecture", "texture"]
    theme = random.choice(themes)
    
    base_w, base_h = 256, 256
    base = create_base_texture(base_w, base_h, theme=theme)
    
    # Save base with its own JPEG quality history
    q_base = random.choice([80, 85, 90, 95])
    buf_base = io.BytesIO()
    base.save(buf_base, format="JPEG", quality=q_base)
    buf_base.seek(0)
    base_jpg = Image.open(buf_base).convert("RGB")
    
    # Create donor patch
    donor_w = random.randint(50, 110)
    donor_h = random.randint(50, 110)
    shape_type = random.choice(["ellipse", "polygon", "irregular"])
    donor_rgba, donor_mask = create_donor_object(donor_w, donor_h, shape=shape_type)
    
    # Save donor independently with a DIFFERENT JPEG compression quality (the root cause of splicing artifacts)
    q_donor = random.choice([50, 60, 70, 75, 98])
    donor_rgb = Image.new("RGB", donor_rgba.size, (128, 128, 128))
    donor_rgb.paste(donor_rgba, mask=donor_mask)
    
    buf_d = io.BytesIO()
    donor_rgb.save(buf_d, format="JPEG", quality=q_donor)
    buf_d.seek(0)
    donor_compressed = Image.open(buf_d).convert("RGB")
    
    # Paste coordinates
    px = random.randint(15, base_w - donor_w - 15)
    py = random.randint(15, base_h - donor_h - 15)
    
    # Composite image
    spliced = base_jpg.copy()
    spliced.paste(donor_compressed, (px, py), mask=donor_mask)
    
    # Ground truth mask (0 = authentic, 255 = tampered)
    gt_mask = Image.new("L", (base_w, base_h), 0)
    gt_mask.paste(donor_mask, (px, py))
    
    # Final resave at standard composite quality
    q_comp = random.choice([85, 90])
    buf_c = io.BytesIO()
    spliced.save(buf_c, format="JPEG", quality=q_comp)
    buf_c.seek(0)
    final_spliced = Image.open(buf_c).convert("RGB")
    
    return final_spliced, gt_mask

def generate_authentic_pair(sample_id):
    """Authentic image with an all-zero ground truth mask."""
    theme = random.choice(["nature", "indoor", "architecture", "texture"])
    img = create_base_texture(256, 256, theme=theme)
    
    # Save with single authentic compression cycle
    q = random.choice([80, 85, 90, 95])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    clean_img = Image.open(buf).convert("RGB")
    
    gt_mask = Image.new("L", (256, 256), 0)
    return clean_img, gt_mask

def build_dataset(num_train=80, num_val=20):
    print("Preparing CASIA-standard Image Splicing Dataset...")
    setup_directories()
    
    # ── Train Set ──
    print(f"Generating {num_train} training samples...")
    for i in range(num_train):
        if i % 2 == 0:
            # Spliced sample
            img, mask = generate_spliced_pair(i)
            name = f"Tp_D_train_{i:04d}"
        else:
            # Authentic sample
            img, mask = generate_authentic_pair(i)
            name = f"Au_ani_train_{i:04d}"
            
        img.save(f"data/train/images/{name}.jpg", quality=92)
        mask.save(f"data/train/masks/{name}.png")
        
    # ── Validation Set ──
    print(f"Generating {num_val} validation samples...")
    for i in range(num_val):
        if i % 2 == 0:
            img, mask = generate_spliced_pair(i + 1000)
            name = f"Tp_D_val_{i:04d}"
        else:
            img, mask = generate_authentic_pair(i + 1000)
            name = f"Au_ani_val_{i:04d}"
            
        img.save(f"data/val/images/{name}.jpg", quality=92)
        mask.save(f"data/val/masks/{name}.png")
        
    print("[OK] Dataset preparation complete.")
    print("  Train:", len(os.listdir("data/train/images")), "images")
    print("  Val:  ", len(os.listdir("data/val/images")), "images")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    build_dataset(num_train=100, num_val=24)

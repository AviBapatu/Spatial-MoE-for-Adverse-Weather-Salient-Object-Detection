#!/usr/bin/env python3
"""Generate qualitative results composite figure from real evaluation outputs."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data", "WXSDO_data")
EVAL = os.path.join(BASE, "..", "evaluation", "best_new_1")

# Pick 3 examples: snow (best), fog (middle), dark (hard)
EXAMPLES = [
    ("0466_snow", "Snow"),
    ("0133_fog",  "Fog"),
    ("0001_dark", "Dark"),
]

COLS = ["Input", "GT", "Prediction", "Abs.\\ Error"]
COL_W = 160  # px per column image
ROW_H = 160  # px per row image
LABEL_H = 22  # px for row label
HEADER_H = 24  # px for column header
GAP = 6  # px between cells
MARGIN = 16

n_rows = len(EXAMPLES)
n_cols = len(COLS)
W = MARGIN + n_cols * COL_W + (n_cols - 1) * GAP + MARGIN
H = MARGIN + HEADER_H + n_rows * ROW_H + (n_rows - 1) * GAP + MARGIN

canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# Try to get a font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
    except:
        font = ImageFont.load_default()
        font_sm = font

# Draw column headers
for c, label in enumerate(COLS):
    x = MARGIN + c * (COL_W + GAP)
    y = MARGIN
    # Use raw string for backslash in LaTeX-style label
    display = label.replace("\\ ", "")
    bbox = draw.textbbox((0, 0), display, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + (COL_W - tw) // 2, y + 2), display, fill=(30, 30, 30), font=font)

# Draw each row
for r, (stem, weather) in enumerate(EXAMPLES):
    row_y = MARGIN + HEADER_H + r * (ROW_H + GAP)

    # Row label
    draw.text((2, row_y + ROW_H // 2 - 6), weather, fill=(60, 60, 60), font=font_sm)

    # Load images
    split = "test_real"  # all examples are from test_real
    input_path = os.path.join(DATA, split, "input", f"{stem}.jpg")
    gt_path = os.path.join(DATA, split, "gt", f"{stem}.jpg")
    pred_path = os.path.join(EVAL, split, "none", f"{stem}.png")

    imgs = []
    for p in [input_path, gt_path, pred_path]:
        if os.path.exists(p):
            img = Image.open(p).convert("L").resize((COL_W, ROW_H), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.float32))
        else:
            print(f"WARNING: missing {p}")
            imgs.append(np.zeros((ROW_H, COL_W), dtype=np.float32))

    input_arr, gt_arr, pred_arr = imgs

    # Compute absolute error
    error_arr = np.abs(pred_arr / 255.0 - gt_arr / 255.0)
    error_vis = (np.clip(error_arr, 0, 1) * 255).astype(np.uint8)

    # Convert to RGB for canvas
    all_imgs = [
        Image.fromarray(input_arr.astype(np.uint8), "L").convert("RGB"),
        Image.fromarray(gt_arr.astype(np.uint8), "L").convert("RGB"),
        Image.fromarray(pred_arr.astype(np.uint8), "L").convert("RGB"),
        Image.fromarray(error_vis, "L").convert("RGB"),
    ]

    # Tint error map red
    err_rgb = np.array(all_imgs[3])
    err_rgb[:, :, 0] = np.clip(err_rgb[:, :, 0].astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    err_rgb[:, :, 1] = (err_rgb[:, :, 1] * 0.4).astype(np.uint8)
    err_rgb[:, :, 2] = (err_rgb[:, :, 2] * 0.4).astype(np.uint8)
    all_imgs[3] = Image.fromarray(err_rgb)

    for c, img in enumerate(all_imgs):
        x = MARGIN + 20 + c * (COL_W + GAP)  # 20px offset for row label
        y = row_y
        canvas.paste(img, (x, y))
        # Thin border
        draw.rectangle([x - 1, y - 1, x + COL_W, y + ROW_H], outline=(180, 180, 180))

out_path = os.path.join(BASE, "fig_qualitative.pdf")
canvas.save(out_path, "PDF", quality=95)
print(f"Saved: {out_path} ({W}x{H})")

# Also save PNG for inspection
png_path = os.path.join(BASE, "fig_qualitative.png")
canvas.save(png_path, "PNG")
print(f"Saved: {png_path}")

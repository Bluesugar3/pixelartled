"""
LED matrix idle animation (no camera): simple blink loop on a 128x32 layout.

Requirements on Raspberry Pi:
- hzeller/rpi-rgb-led-matrix Python bindings installed (module: rgbmatrix)
- Pillow for image loading (pip install pillow)

Notes:
- This assumes your single-eye art files are 64x32 RGBA:
  protogen.png, protogen1.png, protogen2.png, protogen3.png
- Right eye is mirrored from the same art.
- If you have two 64x32 panels chained, consider:
  opts.cols = 64; opts.chain_length = 2
"""

import os
import sys
import time
import random
from typing import List

from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions


# Load assets from the script’s directory by default (override with ASSET_DIR env)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.environ.get("ASSET_DIR", SCRIPT_DIR)


def load_eye_frames(files: List[str]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for f in files:
        if not os.path.exists(f):
            print(f"Missing image: {f}")
            print(f"Working dir: {os.getcwd()}")
            print(f"ASSET_DIR: {ASSET_DIR}")
            sys.exit(1)
        frames.append(Image.open(f).convert("RGBA"))
    # Basic validation
    w, h = frames[0].size
    if (w, h) != (64, 32):
        print(f"Warning: expected 64x32 eye art, got {w}x{h} (continuing)")
    return frames


def make_canvas(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (128, 32))
    canvas.paste(left, (0, 0), left)
    canvas.paste(right, (64, 0), right)
    return canvas


# Damaged columns mitigation: set COL_FIX to 'black' | 'drop_green' | 'reduce_green' | 'off'
BAD_COLUMNS = tuple(range(34, 39))  # inclusive: 34..38
COL_FIX_MODE = os.environ.get("COL_FIX", "drop_green")

def apply_column_filter(img: Image.Image) -> Image.Image:
    if COL_FIX_MODE == 'off' or not BAD_COLUMNS:
        return img.convert("RGB")
    im = img.convert("RGB")
    px = im.load()
    w, h = im.size
    for x in BAD_COLUMNS:
        if 0 <= x < w:
            for y in range(h):
                r, g, b = px[x, y]
                if COL_FIX_MODE == 'black':
                    px[x, y] = (0, 0, 0)
                elif COL_FIX_MODE == 'reduce_green':
                    px[x, y] = (r, max(0, g // 4), b)
                else:  # 'drop_green'
                    px[x, y] = (r, 0, b)
    return im


# Blink timing (tunable via env)
# Shorter BLINK_STEP_SEC -> faster blink animation
BLINK_STEP_SEC = float(os.environ.get("BLINK_STEP_SEC", "0.03"))  # default was 0.065
# Longer interval range between blinks
BLINK_INTERVAL_MIN = float(os.environ.get("BLINK_INTERVAL_MIN", "10.0"))
BLINK_INTERVAL_MAX = float(os.environ.get("BLINK_INTERVAL_MAX", "50.0"))


def main() -> None:
    files = [
        os.path.join(ASSET_DIR, n)
        for n in ["protogen.png", "protogen1.png", "protogen2.png", "protogen3.png"]
    ]
    eye_frames = load_eye_frames(files)  # 0=open, 1..3=blink frames

    # Matrix options
    opts = RGBMatrixOptions()
    opts.rows = 32
    # If you actually have two 64x32 panels chained horizontally, use cols=64 and:
    # opts.chain_length = 2
    opts.cols = 128
    opts.hardware_mapping = os.environ.get("HARDWARE_MAPPING", "adafruit-hat")
    # Optional tweaks
    br = os.environ.get("BRIGHTNESS")
    if br and br.isdigit():
        opts.brightness = max(1, min(100, int(br)))

    matrix = RGBMatrix(options=opts)

    def show_frame(idx: int) -> None:
        left = eye_frames[idx]
        right = eye_frames[idx].transpose(Image.FLIP_LEFT_RIGHT)
        canvas = make_canvas(left, right)
        out = apply_column_filter(canvas)
        matrix.SetImage(out)

    # Blink pattern: hold open, then 1-2-3-2-1-0 quick
    open_idx = 0
    sequence = [1, 2, 3, 2, 1, 0]
    try:
        next_blink = time.time() + random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        show_frame(open_idx)
        while True:
            now = time.time()
            if now >= next_blink:
                for idx in sequence:
                    show_frame(idx)
                    time.sleep(BLINK_STEP_SEC)
                # schedule next blink
                next_blink = time.time() + random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
            else:
                # Idle update cadence
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

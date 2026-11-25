#!/usr/bin/env python3
"""Convert an existing 224x224 WASP detection crop to a DINOv3 input NPZ."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "crop", type=Path,
        help="224x224 uint8 BGR .npy crop, or a 224x224 image (requires Pillow)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dinov3_wasp_input.npz"),
        help="output path (an existing file is preserved)",
    )
    args = parser.parse_args()

    if args.crop.suffix.lower() == ".npy":
        bgr = np.load(args.crop, allow_pickle=False)
        rgb = bgr[..., ::-1]
    else:
        from PIL import Image

        with Image.open(args.crop) as image:
            rgb = np.asarray(image.convert("RGB"))

    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        parser.error(f"expected a 224x224x3 uint8 crop, got {rgb.shape} {rgb.dtype}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    images = np.ascontiguousarray(((rgb_float - mean) / std).transpose(2, 0, 1)[None])
    try:
        with args.output.open("xb") as output:
            np.savez(output, images=images)
    except FileExistsError:
        parser.error(f"output already exists: {args.output}")
    print(f"{args.output.resolve()}: images {images.shape} {images.dtype}")


if __name__ == "__main__":
    main()

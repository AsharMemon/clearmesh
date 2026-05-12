#!/usr/bin/env python3
"""Create a whole-object single-part mask for OmniPart plumbing tests.

OmniPart's CLI expects a part-id mask, commonly an EXR where all channels contain
part ids. This helper creates a conservative all-one mask from an input image so
we can test the command path before adding semantic mask generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-id", type=float, default=1.0)
    args = parser.parse_args()

    image = Image.open(args.image)
    width, height = image.size
    mask = np.full((height, width, 3), args.part_id, dtype=np.float32)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    wrote = False
    error = None
    if output.suffix.lower() == ".exr":
        try:
            import cv2  # type: ignore

            wrote = bool(cv2.imwrite(str(output), mask))
            if not wrote:
                error = "cv2.imwrite returned false"
        except Exception as exc:  # noqa: BLE001 - report fallback reason.
            error = f"{type(exc).__name__}: {exc}"

    if not wrote:
        fallback = output.with_suffix(".npy")
        np.save(fallback, mask)
        output = fallback

    report = {
        "image": str(Path(args.image).expanduser()),
        "mask": str(output),
        "part_id": args.part_id,
        "shape": list(mask.shape),
        "exr_error": error,
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

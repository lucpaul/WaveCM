#!/usr/bin/env python3
"""Create absolute residual maps between mean predictions and ground-truth HR images.

Example:
    python src/utils/create_residuals.py \
      --hr-folder /path/to/hr_200 \
      --mean-folder /path/to/uncertainty_eval_xxx/mean_sr

Outputs are written to:
    /path/to/uncertainty_eval_xxx/residuals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


VALID_EXTS = {".tif", ".tiff"}


def _is_tif(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS and not path.name.startswith(".")


def _collect_tifs(folder: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(folder.iterdir()):
        if _is_tif(path):
            files[path.name] = path
    return files


def _spatial_hw(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return arr.shape[0], arr.shape[1]
    if arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr.shape[1], arr.shape[2]
        return arr.shape[0], arr.shape[1]
    raise ValueError(f"Unsupported image ndim={arr.ndim}; expected 2D or 3D TIFF.")


def _crop_top_left(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:target_h, :target_w]
    if arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr[:, :target_h, :target_w]
        return arr[:target_h, :target_w, ...]
    raise ValueError(f"Unsupported image ndim={arr.ndim}; expected 2D or 3D TIFF.")


def create_residuals(hr_folder: Path, mean_folder: Path, shape_policy: str = "crop-top-left") -> Path:
    if not hr_folder.is_dir():
        raise FileNotFoundError(f"HR folder not found: {hr_folder}")
    if not mean_folder.is_dir():
        raise FileNotFoundError(f"Mean folder not found: {mean_folder}")

    residuals_folder = mean_folder.parent / "residuals"
    residuals_folder.mkdir(parents=True, exist_ok=True)

    hr_files = _collect_tifs(hr_folder)
    mean_files = _collect_tifs(mean_folder)

    if not hr_files:
        raise ValueError(f"No TIFF files found in HR folder: {hr_folder}")
    if not mean_files:
        raise ValueError(f"No TIFF files found in mean folder: {mean_folder}")

    common_names = sorted(set(hr_files.keys()) & set(mean_files.keys()))
    missing_in_hr = sorted(set(mean_files.keys()) - set(hr_files.keys()))
    missing_in_mean = sorted(set(hr_files.keys()) - set(mean_files.keys()))

    if not common_names:
        raise ValueError(
            "No matching TIFF filenames found between folders. "
            f"HR folder: {hr_folder}, mean folder: {mean_folder}"
        )

    if missing_in_hr:
        print(f"Warning: {len(missing_in_hr)} prediction files have no matching HR file.")
        print("  First few:", ", ".join(missing_in_hr[:5]))
    if missing_in_mean:
        print(f"Warning: {len(missing_in_mean)} HR files have no matching prediction file.")
        print("  First few:", ", ".join(missing_in_mean[:5]))

    written = 0
    aligned = 0
    for name in common_names:
        hr_path = hr_files[name]
        mean_path = mean_files[name]

        hr = tifffile.imread(str(hr_path))
        mean = tifffile.imread(str(mean_path))

        if hr.shape != mean.shape:
            if shape_policy == "strict":
                raise ValueError(
                    f"Shape mismatch for {name}: HR {hr.shape} vs mean {mean.shape}"
                )

            hr_h, hr_w = _spatial_hw(hr)
            mean_h, mean_w = _spatial_hw(mean)
            target_h = min(hr_h, mean_h)
            target_w = min(hr_w, mean_w)

            if target_h <= 0 or target_w <= 0:
                raise ValueError(
                    f"Invalid spatial sizes for {name}: HR {hr.shape} vs mean {mean.shape}"
                )

            hr = _crop_top_left(hr, target_h, target_w)
            mean = _crop_top_left(mean, target_h, target_w)
            aligned += 1

            if hr.shape != mean.shape:
                raise ValueError(
                    f"Unable to align shapes for {name}: HR {hr.shape} vs mean {mean.shape}"
                )

        residual = np.abs(mean.astype(np.float32) - hr.astype(np.float32))
        out_path = residuals_folder / name
        tifffile.imwrite(str(out_path), residual)
        written += 1

    print(f"Processed {written} residual files.")
    if aligned:
        print(f"Auto-aligned {aligned} files via top-left crop to common spatial size.")
    print(f"Residuals saved to: {residuals_folder}")
    return residuals_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Create absolute residual TIFFs from mean predictions and HR targets.")
    parser.add_argument("--hr-folder", required=True, help="Path to HR ground-truth folder.")
    parser.add_argument("--mean-folder", required=True, help="Path to mean_sr prediction folder.")
    parser.add_argument(
        "--shape-policy",
        default="crop-top-left",
        choices=["crop-top-left", "strict"],
        help=(
            "How to handle shape mismatch: 'crop-top-left' (default) crops both images "
            "to a shared spatial size; 'strict' raises an error."
        ),
    )
    args = parser.parse_args()

    hr_folder = Path(args.hr_folder).expanduser().resolve()
    mean_folder = Path(args.mean_folder).expanduser().resolve()

    create_residuals(hr_folder, mean_folder, shape_policy=args.shape_policy)


if __name__ == "__main__":
    main()

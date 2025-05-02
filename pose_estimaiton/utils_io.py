"""
utils_io.py
===========

Light-weight helpers for reading / writing JSON and camera-parameter files.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def load_json(path: str | Path) -> Dict:
    """Return a Python dict parsed from *path*."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict | List, path: str | Path, indent: int = 4) -> None:
    """Write *data* to *path* in UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)
    print(f"[IO] JSON saved → {path}")


# --------------------------------------------------------------------------- #
# Camera-parameter loaders
# --------------------------------------------------------------------------- #
def load_intrinsics(file_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the `cameraX_intrinsics.dat` file produced by your calibration script.

    Returns
    -------
    K        : (3, 3) ndarray  – intrinsic matrix
    distCoef : (N,)  ndarray   – distortion coefficients
    """
    lines = Path(file_path).read_text().strip().splitlines()
    K = np.array([list(map(float, ln.split())) for ln in lines[1:4]])
    dist = np.array(list(map(float, lines[5].split())))
    return K, dist


def load_extrinsics(file_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the `cameraX_rot_trans.dat` file (R / T in plain text).

    Returns
    -------
    R : (3, 3) ndarray
    t : (3, 1) ndarray
    """
    r_rows, t_rows, mode = [], [], None
    for ln in Path(file_path).read_text().splitlines():
        if ln.startswith("R"): mode = "R"; continue
        if ln.startswith("T"): mode = "T"; continue
        (r_rows if mode == "R" else t_rows).append(list(map(float, ln.split())))
    R = np.array(r_rows)
    t = np.array(t_rows).reshape(3, 1)
    return R, t

"""
triangulate3d.py
================

Convert paired 2-D landmarks into 3-D coordinates via linear
triangulation and calibrated camera parameters.

Example
-------
$ python triangulate3d.py --pose pose2d.json --out pose3d.json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from typing import List

from utils_io import load_json, save_json, load_intrinsics, load_extrinsics


# ──────────────────────────────────────────────────────────────────────────── #
def build_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return 3×4 projection matrix P = K · [R | t]."""
    Rt = np.hstack((R, t))
    return K @ Rt


def triangulate(P0: np.ndarray, P1: np.ndarray,
                pts0: np.ndarray, pts1: np.ndarray) -> List[List[float]]:
    """Triangulate each pair of points (Nx2, Nx2) into (x, y, z)."""
    pts_3d = []
    for p0, p1 in zip(pts0, pts1):
        X4 = cv.triangulatePoints(P0, P1, p0.reshape(2, 1), p1.reshape(2, 1))
        X3 = (X4[:3] / X4[3]).ravel()
        pts_3d.append(X3.tolist())
    return pts_3d


# ──────────────────────────────────────────────────────────────────────────── #
def process(pose_json: Path, out_json: Path) -> None:
    """Main driver: read 2-D JSON, load camera params, save 3-D JSON."""
    data = load_json(pose_json)

    # Camera calibration files (produced earlier)
    K0, _ = load_intrinsics("camera_parameters/camera0_intrinsics.dat")
    K1, _ = load_intrinsics("camera_parameters/camera1_intrinsics.dat")
    R0, t0 = np.eye(3), np.zeros((3, 1))
    R1, t1 = load_extrinsics("camera_parameters/camera1_rot_trans.dat")

    P0 = build_projection(K0, R0, t0)
    P1 = build_projection(K1, R1, t1)

    pts3d_all = []
    for fr0, fr1 in zip(data["video_0"], data["video_1"]):
        pts0 = np.array([[lm["x"], lm["y"]] for lm in fr0["landmarks"]], dtype=np.float32)
        pts1 = np.array([[lm["x"], lm["y"]] for lm in fr1["landmarks"]], dtype=np.float32)
        pts3d_all.extend(triangulate(P0, P1, pts0, pts1))

    save_json({"points_3d": pts3d_all}, out_json)
    print(f"[3-D] saved → {out_json}")


# ──────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="Input 2-D JSON")
    ap.add_argument("--out", default="pose3d.json", help="Output 3-D JSON")
    args = ap.parse_args()
    process(Path(args.pose), Path(args.out))

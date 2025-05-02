#!/usr/bin/env python3
"""
Stereo Camera Calibration
=========================

Detects a chessboard pattern in paired images, computes intrinsic
parameters for each camera, performs stereo calibration to obtain the
relative pose (R, T), and stores all results to *stereo_calibration_data.npz*.

The script assumes your image pairs live in a directory called
``frames_pair`` and are named::

    camera0_XX.png
    camera1_XX.png

where *XX* is the running index (01, 02, …).

You can adapt paths, pattern size, and image size via the constants at
the top of the file.
"""

from __future__ import annotations

import os
import glob
from typing import List, Tuple

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------#
# User configuration
# -----------------------------------------------------------------------------#
CHESSBOARD_SIZE: Tuple[int, int] = (9, 6)          # inner corners (cols, rows)
FRAME_SIZE: Tuple[int, int] = (640, 480)           # width, height in px
PAIR_FOLDER: str = "frames_pair"                   # where image pairs live
LEFT_PREFIX: str = "camera0_"                      # file-name prefix (left)
RIGHT_PREFIX: str = "camera1_"                     # file-name prefix (right)
OUTPUT_FILE: str = "stereo_calibration_data.npz"   # result archive


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#
def _object_points() -> NDArray[np.floating]:
    """Generate the canonical Z = 0 grid for a planar chessboard."""
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : CHESSBOARD_SIZE[0],
        0 : CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    return objp


def _collect_image_pairs() -> List[Tuple[str, str]]:
    """Return sorted (left, right) filename pairs available in *PAIR_FOLDER*."""
    left_imgs = sorted(
        glob.glob(os.path.join(PAIR_FOLDER, f"{LEFT_PREFIX}*"))
    )
    right_imgs = sorted(
        glob.glob(os.path.join(PAIR_FOLDER, f"{RIGHT_PREFIX}*"))
    )
    pairs = list(zip(left_imgs, right_imgs))
    if not pairs:
        raise FileNotFoundError("No matching image pairs found in "
                                f"folder '{PAIR_FOLDER}'.")
    return pairs


def _find_corners(
    img: NDArray[np.uint8], criteria: Tuple[int, int, float]
) -> Tuple[bool, NDArray[np.floating]]:
    """
    Detect chessboard corners and refine them to sub-pixel accuracy.
    Returns ``(found, corners)`` where *corners* is ``None`` if not found.
    """
    found, corners = cv.findChessboardCorners(img, CHESSBOARD_SIZE, None)
    if found:
        corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def _calibrate_single(
    objpoints: List[NDArray[np.floating]],
    imgpoints: List[NDArray[np.floating]],
) -> Tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    """Wrapper for ``cv.calibrateCamera`` with nicer return order."""
    rms, K, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, FRAME_SIZE, None, None
    )
    return rms, K, dist


# -----------------------------------------------------------------------------#
# Main routine
# -----------------------------------------------------------------------------#
def main() -> None:
    # Termination criteria for corner refinement
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    # Storage containers
    objp_template = _object_points()
    objpoints: List[NDArray[np.floating]] = []
    imgpoints_left: List[NDArray[np.floating]] = []
    imgpoints_right: List[NDArray[np.floating]] = []

    # Iterate over paired images
    for left_path, right_path in _collect_image_pairs():
        left_img = cv.imread(left_path, cv.IMREAD_GRAYSCALE)
        right_img = cv.imread(right_path, cv.IMREAD_GRAYSCALE)

        ret_l, corners_l = _find_corners(left_img, criteria)
        ret_r, corners_r = _find_corners(right_img, criteria)

        if ret_l and ret_r:
            objpoints.append(objp_template)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)

    if not objpoints:
        raise RuntimeError("No valid chessboard detections – calibration aborted.")

    # ---- Mono calibration --------------------------------------------------- #
    rms_l, K_l, d_l = _calibrate_single(objpoints, imgpoints_left)
    rms_r, K_r, d_r = _calibrate_single(objpoints, imgpoints_right)
    print(f"[LEFT ] RMS reprojection error: {rms_l:.4f}")
    print(f"[RIGHT] RMS reprojection error: {rms_r:.4f}")

    # ---- Stereo calibration ------------------------------------------------- #
    stereo_rms, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_l,
        d_l,
        K_r,
        d_r,
        FRAME_SIZE,
        flags=cv.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    print(f"[STEREO] RMS reprojection error: {stereo_rms:.4f}")
    print("[STEREO] Rotation matrix R:\n", R)
    print("[STEREO] Translation vector T:\n", T.ravel())

    # ---- Persist results ---------------------------------------------------- #
    np.savez(
        OUTPUT_FILE,
        camera_matrix_left=K_l,
        dist_coeffs_left=d_l,
        camera_matrix_right=K_r,
        dist_coeffs_right=d_r,
        R=R,
        T=T,
        image_size=np.array(FRAME_SIZE),
    )
    print(f"[OK] Calibration data saved to '{OUTPUT_FILE}'")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()

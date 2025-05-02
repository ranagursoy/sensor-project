"""
pose2d_extractor.py
===================

Extract 2-D landmarks from *two* synchronized videos (or dual-lens footage)
and store per-frame landmarks in a single JSON file.

Example
-------
$ python pose2d_extractor.py --v0 cam0.mp4 --v1 cam1.mp4 --out pose2d.json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2 as cv
import mediapipe as mp
from typing import List, Dict

from utils_io import save_json


# ──────────────────────────────────────────────────────────────────────────── #
# Mediapipe pose helper
# ──────────────────────────────────────────────────────────────────────────── #
def init_pose() -> mp.solutions.pose.Pose:
    """Return a configured MediaPipe Pose model."""
    return mp.solutions.pose.Pose(model_complexity=2,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


def detect_landmarks(frame, pose) -> List[Dict] | None:
    """Run pose inference on *frame* and return landmark dicts or *None*."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in res.pose_landmarks.landmark]


# ──────────────────────────────────────────────────────────────────────────── #
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────── #
def process(video0: Path, video1: Path, out_json: Path) -> None:
    """Extract landmarks from *video0* and *video1* and dump to *out_json*."""
    cap0, cap1 = (cv.VideoCapture(str(video0)), cv.VideoCapture(str(video1)))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Video(s) could not be opened")

    pose = init_pose()
    output: Dict[str, List] = {"video_0": [], "video_1": []}
    frame_idx = 0

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            break

        # Split stereo frame in half if necessary
        f0_half = f0[:, : f0.shape[1] // 2]
        f1_half = f1[:, f1.shape[1] // 2:]

        for cam_id, frm in enumerate((f0_half, f1_half)):
            lm = detect_landmarks(frm, pose)
            if lm:
                output[f"video_{cam_id}"].append({"frame": frame_idx, "landmarks": lm})

        frame_idx += 1
        if cv.waitKey(1) & 0xFF == 27:   # ESC to abort early
            break

    cap0.release(), cap1.release(), cv.destroyAllWindows()
    save_json(output, out_json)


# ──────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True, help="Left-camera / video-0 file")
    ap.add_argument("--v1", required=True, help="Right-camera / video-1 file")
    ap.add_argument("--out", default="pose2d.json", help="Output JSON file")
    args = ap.parse_args()
    process(Path(args.v0), Path(args.v1), Path(args.out))

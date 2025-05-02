"""
viewer3d.py
===========

Matplotlib animation that steps through timestamps inside a 3-D
pose JSON and shows skeleton + key-point scatter.

Usage
-----
$ python viewer3d.py pose3d.json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                             # noqa: F401
from typing import List

from utils_io import load_json


# Skeleton edges (33-keypoints -> rough humanoid)
_CONNECTIONS: List[tuple[int, int]] = [
    (0, 11), (0, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28),
]


def plot_skeleton(ax, pts3d: np.ndarray) -> None:
    """Draw points + bones on *ax*."""
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c="blue", s=20)
    for i, j in _CONNECTIONS:
        ax.plot(*zip(pts3d[i], pts3d[j]), c="red")


def main(json_path: Path) -> None:
    data = load_json(json_path)
    pts3d = np.array(data["points_3d"]).reshape(-1, 33, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for frame_pts in pts3d:
        ax.clear()
        plot_skeleton(ax, frame_pts)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z",
               title="3-D Pose – press any key for next frame")
        plt.draw()
        plt.waitforbuttonpress()

    print("[VIEW] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="3-D pose JSON")
    main(Path(parser.parse_args().json))

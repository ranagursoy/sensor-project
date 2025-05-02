"""
Direct-Linear-Transform (DLT) triangulation
==========================================

A minimal helper for recovering a 3-D point X from its projections in two
pinhole cameras with known 3 × 4 projection matrices *P1* and *P2*.

Example
-------
>>> from dlt import triangulate_dlt
>>> X = triangulate_dlt(P1, P2, (u1, v1), (u2, v2))
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def triangulate_dlt(
    P1: NDArray[np.floating], P2: NDArray[np.floating],
    p1: tuple[float, float],   p2: tuple[float, float]
) -> NDArray[np.floating]:
    """
    Parameters
    ----------
    P1, P2 : (3, 4) ndarray
        Projection matrices of camera-1 and camera-2.
    p1, p2 : (u, v) tuple
        2-D image coordinates of the same world point in each view.

    Returns
    -------
    ndarray, shape (3,)
        Euclidean coordinates **X** of the triangulated 3-D point.
    """
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1],
    ])

    # Solve A X = 0 in the least-squares sense via SVD
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]                        # last row is the solution
    return X_h[:3] / X_h[3]             # de-homogenise

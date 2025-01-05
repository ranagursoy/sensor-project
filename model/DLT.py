import numpy as np

def DLT(P1, P2, point1, point2):
    """
    P1: Kamera 1 projeksiyon matrisi
    P2: Kamera 2 projeksiyon matrisi
    point1: Kamera 1'den 2D nokta (u1, v1)
    point2: Kamera 2'den 2D nokta (u2, v2)
    """
    A = np.array([
        point1[0] * P1[2, :] - P1[0, :],
        point1[1] * P1[2, :] - P1[1, :],
        point2[0] * P2[2, :] - P2[0, :],
        point2[1] * P2[2, :] - P2[1, :]
    ])
    
    # En küçük kareler yöntemi ile çözüm
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    
    # Homojen koordinattan 3D'ye dönüşüm
    return X[:3] / X[3]

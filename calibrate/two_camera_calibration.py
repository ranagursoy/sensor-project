import cv2
import numpy as np
import glob
import os

# Satranç tahtası köşe sayısı (iç köşeler)
chessboard_size = (9, 6)
frame_size = (640, 480)  # Görüntü çözünürlüğü

# 3D Noktaları oluşturma (Z = 0 düzleminde)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Nokta tutucular
objpoints = []  # Gerçek dünya noktaları
imgpoints_left = []  # Sol kamera noktaları
imgpoints_right = []  # Sağ kamera noktaları

# Sol ve sağ görüntülerin yolları
frames_path = "frames_pair"  # frames_pair klasörü
left_images = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.startswith('camera0_')])
right_images = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.startswith('camera1_')])

# Görüntü çiftlerini döngüyle işleme
for left_path, right_path in zip(left_images, right_images):
    # Sol ve sağ görüntüleri yükle
    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    # Satranç tahtasını algıla
    ret_left, corners_left = cv2.findChessboardCorners(left_img, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(right_img, chessboard_size, None)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        # Köşeleri alt piksellerle iyileştir
        corners_left = cv2.cornerSubPix(left_img, corners_left, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_right = cv2.cornerSubPix(right_img, corners_right, (11, 11), (-1, -1),
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

# Sol ve sağ kamera için kalibrasyon
ret_left, camera_matrix_left, dist_coeffs_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, frame_size, None, None)
ret_right, camera_matrix_right, dist_coeffs_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, frame_size, None, None)

# Stereo kalibrasyon
ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right, frame_size, flags=cv2.CALIB_FIX_INTRINSIC)

# Kalibrasyon sonuçlarını kaydetme
np.savez('stereo_calibration_data.npz',
         camera_matrix_left=camera_matrix_left,
         dist_coeffs_left=dist_coeffs_left,
         camera_matrix_right=camera_matrix_right,
         dist_coeffs_right=dist_coeffs_right,
         R=R, T=T, image_size=frame_size)

print("Stereo kalibrasyon tamamlandı ve stereo_calibration_data.npz dosyasına kaydedildi.")

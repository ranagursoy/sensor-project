import numpy as np
import json
import cv2


# Kamera içsel parametrelerini okuma
def load_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        intrinsic = np.array([list(map(float, line.split())) for line in lines[1:4]])
        distortion = np.array(list(map(float, lines[5].split())))
    return intrinsic, distortion


# Kamera dönüş ve öteleme matrisini okuma
# Kamera dönüş ve öteleme matrisini okuma
def load_extrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        r_lines = []
        t_lines = []
        r_flag = False
        t_flag = False
        
        for line in lines:
            if line.startswith("R:"):
                r_flag = True
                t_flag = False
                continue
            elif line.startswith("T:"):
                t_flag = True
                r_flag = False
                continue
            
            if r_flag:
                r_lines.append(list(map(float, line.split())))
            elif t_flag:
                t_lines.append(float(line.strip()))

        rotation = np.array(r_lines)
        translation = np.array(t_lines).reshape(3, 1)
        
    return rotation, translation



# Kamera parametrelerini yükleme
camera0_intrinsics, camera0_distortion = load_intrinsics("camera_parameters/camera0_intrinsics.dat")
camera1_intrinsics, camera1_distortion = load_intrinsics("camera_parameters/camera1_intrinsics.dat")

camera0_rotation, camera0_translation = load_extrinsics("camera_parameters/camera0_rot_trans.dat")
camera1_rotation, camera1_translation = load_extrinsics("camera_parameters/camera1_rot_trans.dat")

# Kamera matrisleri
camera0_projection = np.hstack((np.eye(3), np.zeros((3, 1))))
camera1_projection = np.hstack((camera1_rotation, camera1_translation.reshape(-1, 1)))

# Kamera matrislerini içsel parametrelerle çarpma
P0 = np.dot(camera0_intrinsics, camera0_projection)
P1 = np.dot(camera1_intrinsics, camera1_projection)

# JSON dosyasını yükleme
with open('pose_data.json', 'r') as f:
    pose_data = json.load(f)

# 3D noktaları hesaplama
points_3d = []

for frame_data_1, frame_data_2 in zip(pose_data['video_1'], pose_data['video_2']):
    landmarks_1 = np.array([[lm['x'], lm['y']] for lm in frame_data_1['landmarks']])
    landmarks_2 = np.array([[lm['x'], lm['y']] for lm in frame_data_2['landmarks']])

    for lm1, lm2 in zip(landmarks_1, landmarks_2):
        point_4d = cv2.triangulatePoints(P0, P1, lm1.T, lm2.T)
        point_3d = point_4d[:3] / point_4d[3]  # Homojen koordinatları normalize et
        points_3d.append(point_3d.flatten().tolist())  # .tolist() ekledik

# 3D noktaları JSON olarak kaydetme
output_3d_json = {
    "points_3d": points_3d
}

with open('pose_3d_data.json', 'w') as f:
    json.dump(output_3d_json, f, indent=4)  # Artık hata vermeyecek

print("3D pose data saved to pose_3d_data.json")

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# JSON dosyasını yükleme
json_path = "./output-2.json"

# Bağlantı çiftleri (34 anahtar nokta için)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 26), (3, 4), (3, 11), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), 
    (7, 10), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17), (18, 19), 
    (19, 20), (20, 21), (20, 32), (22, 23), (23, 24), (24, 25), (24, 33), (26, 27), (27, 28), (28, 29), 
    (30, 31), (27, 30), (0, 22), (0, 18)
]

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Timestamp'i insan tarafından okunabilir hale dönüştür

def timestamp_to_datetime(ts):
    return datetime.fromtimestamp(ts / 1e9).strftime('%Y-%m-%d %H:%M:%S')


def plot_3d_keypoints(json_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Timestamp'leri sıraya göre işle
    sorted_timestamps = sorted(json_data.keys())

    for timestamp in sorted_timestamps:
        ax.clear()  # Önceki frame'i temizle
        body_info = json_data[timestamp]
        human_readable_time = timestamp_to_datetime(body_info['timestamp'])
        
        for body in body_info['body_list']:
            keypoints = np.array(body['keypoint'])
            if keypoints.shape[0] > 0:
                x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
                
                # Anahtar noktaları 3D olarak göster
                ax.scatter(x, y, z, s=20, c='blue', marker='o')
                
                # Bağlantıları çiz
                for conn in connections:
                    kp1_idx, kp2_idx = conn
                    kp1 = keypoints[kp1_idx]
                    kp2 = keypoints[kp2_idx]
                    ax.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], [kp1[2], kp2[2]], color='red')

        ax.set_xlabel('X Ekseni (metre)')
        ax.set_ylabel('Y Ekseni (metre)')
        ax.set_zlabel('Z Ekseni (metre)')
        ax.set_title(f'3D Vücut Anahtar Noktaları - {human_readable_time}')

        plt.draw()
        plt.pause(0.5)  # Her frame arasında 0.5 saniye bekler
        plt.waitforbuttonpress()  # Bir tuşa basıldığında bir sonraki frame'e geçer


def main():
    json_data = load_json_data(json_path)
    plot_3d_keypoints(json_data)

if __name__ == "__main__":
    main()

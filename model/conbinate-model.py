import cv2
import mediapipe as mp
import json
import numpy as np


# Sabitler (Constants)
SOURCE_VIDEO_PATH_1 = "kayıtlar/kamera-rana-0-yeni.mp4"
SOURCE_VIDEO_PATH_2 = "kayıtlar/kamera-rana-0-yeni.mp4"
OUTPUT_VIDEO_PATH_1 = 'output_half_pose_video_1.mp4'
OUTPUT_VIDEO_PATH_2 = 'output_half_pose_video_2.mp4'
OUTPUT_JSON_PATH = 'pose_data.json'

FRAME_WIDTH_DIVISOR = 2
FRAME_HEIGHT_DIVISOR = 2
VIDEO_FPS = 30.0

# Mediapipe Pose ayarları
POSE_MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# Pose modelini başlat
def initialize_pose_model(static_image_mode=False):
    return mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=POSE_MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )


# Video akışını açma
def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video {video_path} cannot be opened.")
        exit()
    return cap


# Video kaydedici başlatma
def initialize_video_writer(output_path, frame_width, frame_height, fps=VIDEO_FPS):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# 2D pose tespiti yapan fonksiyon
def detect_2d_pose(frame, pose_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb_frame)

    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )
        return result.pose_landmarks
    return None


# Landmark'ları JSON formatına dönüştürme
def convert_landmarks_to_json(landmarks, frame_count):
    return {
        "frame": frame_count,
        "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks.landmark]
    }


# 2D poz verilerini çıkar ve kaydet
def process_pose_from_video(cap, pose_model, frame_width, frame_height, side='left'):
    frame_count = 0
    pose_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (frame_width, frame_height))

        half_frame = frame[:, :frame.shape[1] // 2] if side == 'left' else frame[:, frame.shape[1] // 2:]

        pose_landmarks = detect_2d_pose(half_frame, pose_model)
        if pose_landmarks:
            pose_data.append(convert_landmarks_to_json(pose_landmarks, frame_count))

        cv2.imshow(f'Pose Detection ({side.capitalize()} Frame)', half_frame)
        yield half_frame, pose_data

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# 3D nokta dönüşümü için triangulation işlemi
def triangulate_3d_points(P0, P1, landmarks_1, landmarks_2):
    points_3d = []

    for lm1, lm2 in zip(landmarks_1, landmarks_2):
        point_4d = cv2.triangulatePoints(P0, P1, lm1.T, lm2.T)
        point_3d = point_4d[:3] / point_4d[3]  # Homojen koordinatları normalize et
        points_3d.append(point_3d.flatten().tolist())

    return points_3d


# JSON dosyasına pose verilerini kaydet
def save_pose_data_to_json(pose_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=4)
    print(f"Pose data saved to {output_path}")


# Ana çalışma akışı
def main():
    # Pose modeli başlat
    pose_model = initialize_pose_model()

    # Video akışlarını başlat
    cap1 = initialize_video_capture(SOURCE_VIDEO_PATH_1)
    cap2 = initialize_video_capture(SOURCE_VIDEO_PATH_2)

    # Çözünürlük ayarları
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) // FRAME_WIDTH_DIVISOR
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) // FRAME_HEIGHT_DIVISOR

    # Video kaydedicileri başlat
    out1 = initialize_video_writer(OUTPUT_VIDEO_PATH_1, frame_width, frame_height)
    out2 = initialize_video_writer(OUTPUT_VIDEO_PATH_2, frame_width, frame_height)

    # Pose verilerini saklamak için JSON yapısı
    pose_data = {
        "video_1": [],
        "video_2": []
    }

    # Video 1 için 2D poz çıkarımı
    for half_frame1, video_1_pose_data in process_pose_from_video(
            cap1, pose_model, frame_width, frame_height, side='left'):
        out1.write(half_frame1)
        pose_data['video_1'].extend(video_1_pose_data)

    # Video 2 için 2D poz çıkarımı
    for half_frame2, video_2_pose_data in process_pose_from_video(
            cap2, pose_model, frame_width, frame_height, side='right'):
        out2.write(half_frame2)
        pose_data['video_2'].extend(video_2_pose_data)

    # Kaynakları serbest bırak
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

    # Pose verilerini JSON olarak kaydet
    save_pose_data_to_json(pose_data, OUTPUT_JSON_PATH)


if __name__ == '__main__':
    main()

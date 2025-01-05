import cv2
import mediapipe as mp
import json


# Pose modelini başlat
def initialize_pose_model():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


# Video akışını açma ve kontrol etme
def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video {video_path} cannot be opened.")
        exit()
    return cap


# Video kaydedici başlatma
def initialize_video_writer(output_path, frame_width, frame_height, fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# Pose tespiti yap ve landmark verilerini çıkar
def process_pose_from_video(cap, pose_model, frame_width, frame_height, side='left'):
    frame_count = 0
    pose_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Sol veya sağ yarım kareyi al
        half_frame = frame[:, :frame.shape[1] // 2] if side == 'left' else frame[:, frame.shape[1] // 2:]

        # RGB formatına çevir ve poz tespiti yap
        rgb_frame = cv2.cvtColor(half_frame, cv2.COLOR_BGR2RGB)
        result = pose_model.process(rgb_frame)

        # Landmark verilerini çıkar ve çiz
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                half_frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
            pose_data.append({
                "frame": frame_count,
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z} for lm in result.pose_landmarks.landmark
                ]
            })

        # Frame'i göster
        cv2.imshow(f'Pose Detection ({side.capitalize()} Frame)', half_frame)

        # Kaydet
        yield half_frame, pose_data

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# JSON dosyasına poz verilerini kaydet
def save_pose_data_to_json(pose_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(pose_data, f, indent=4)
    print(f"Pose data saved to {output_path}")


def main():
    # Kaynak videolar
    SOURCE_VIDEO_PATH_1 = "kayıtlar/kamera-rana-0-yeni.mp4"
    SOURCE_VIDEO_PATH_2 = "kayıtlar/kamera-rana-0-yeni.mp4"

    # Çıkış video yolları
    output_video_path_1 = 'output_half_pose_video_1.mp4'
    output_video_path_2 = 'output_half_pose_video_2.mp4'
    output_json_path = 'pose_data.json'

    # Pose modeli başlat
    pose_model = initialize_pose_model()

    # Videoları başlat
    cap1 = initialize_video_capture(SOURCE_VIDEO_PATH_1)
    cap2 = initialize_video_capture(SOURCE_VIDEO_PATH_2)

    # Çözünürlük ayarları
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    # Video kaydedicileri başlat
    out1 = initialize_video_writer(output_video_path_1, frame_width, frame_height)
    out2 = initialize_video_writer(output_video_path_2, frame_width, frame_height)

    # Pose verilerini saklamak için JSON formatı
    pose_data = {
        "video_1": [],
        "video_2": []
    }

    # Video 1 için pose çıkarımı
    for half_frame1, video_1_pose_data in process_pose_from_video(
            cap1, pose_model, frame_width, frame_height, side='left'):
        out1.write(half_frame1)
        pose_data['video_1'].extend(video_1_pose_data)

    # Video 2 için pose çıkarımı
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
    save_pose_data_to_json(pose_data, output_json_path)


if __name__ == '__main__':
    main()

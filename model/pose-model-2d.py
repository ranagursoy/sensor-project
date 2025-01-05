import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

SOURCE_VIDEO_PATH_1 = "kayıtlar/kamera-rana-0-yeni.mp4"
SOURCE_VIDEO_PATH_2 = "kayıtlar/kamera-rana-0-yeni.mp4"

cap1 = cv2.VideoCapture(SOURCE_VIDEO_PATH_1)
cap2 = cv2.VideoCapture(SOURCE_VIDEO_PATH_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: One or both videos cannot be opened.")
    exit()

frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

output_video_path_1 = 'output_half_pose_video_1.mp4'
output_video_path_2 = 'output_half_pose_video_2.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter(output_video_path_1, fourcc, 30.0, (frame_width, frame_height))
out2 = cv2.VideoWriter(output_video_path_2, fourcc, 30.0, (frame_width, frame_height))

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_data = {
    "video_1": [],
    "video_2": []
}

frame_count = 0

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    frame_count += 1

    frame1 = cv2.resize(frame1, (frame_width, frame_height))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * 1.5), int(frame2.shape[0] * 1.5)))

    half_frame1 = frame1[:, :frame1.shape[1] // 2]
    half_frame2 = frame2[:, frame2.shape[1] // 2:]

    rgb_frame1 = cv2.cvtColor(half_frame1, cv2.COLOR_BGR2RGB)
    rgb_frame2 = cv2.cvtColor(half_frame2, cv2.COLOR_BGR2RGB)

    result1 = pose_model.process(rgb_frame1)
    result2 = pose_model.process(rgb_frame2)

    if result1.pose_landmarks:
        mp_drawing.draw_landmarks(half_frame1, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_data["video_1"].append({
            "frame": frame_count,
            "landmarks": [{ "x": lm.x, "y": lm.y, "z": lm.z } for lm in result1.pose_landmarks.landmark]
        })

    if result2.pose_landmarks:
        mp_drawing.draw_landmarks(half_frame2, result2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_data["video_2"].append({
            "frame": frame_count,
            "landmarks": [{ "x": lm.x, "y": lm.y, "z": lm.z } for lm in result2.pose_landmarks.landmark]
        })

    cv2.imshow('Pose Detection (Video 1)', half_frame1)
    cv2.imshow('Pose Detection (Video 2)', half_frame2)

    out1.write(half_frame1)
    out2.write(half_frame2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

output_json_path = 'pose_data.json'
with open(output_json_path, 'w') as f:
    json.dump(pose_data, f, indent=4)

print(f"Pose data saved to {output_json_path}")

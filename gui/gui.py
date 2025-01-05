import sys
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
                             QScrollArea, QDialog, QTextEdit, QLineEdit, QFileDialog, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mediapipe as mp

# MediaPipe pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Keypoint names from Mediapipe
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

# Load JSON data
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)

class KeypointDialog(QDialog):
    def __init__(self, keypoint_name):
        super().__init__()
        self.setWindowTitle(f"Keypoint: {keypoint_name}")
        self.setGeometry(100, 100, 400, 400)

        self.keypoint_name = keypoint_name
        self.x = 0
        self.y = 0
        self.z = 0

        self.layout = QVBoxLayout()

        # 3D plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.layout.addWidget(self.canvas)

        # Coordinates
        self.coordinates = QTextEdit()
        self.coordinates.setReadOnly(True)
        self.layout.addWidget(self.coordinates)

        self.setLayout(self.layout)

    def update_data(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        # Update plot
        self.canvas.axes.clear()
        self.canvas.axes.scatter(x, y, z, c='blue', label=self.keypoint_name)
        self.canvas.axes.set_xlabel('X')
        self.canvas.axes.set_ylabel('Y')
        self.canvas.axes.set_zlabel('Z')
        self.canvas.axes.legend()
        self.canvas.draw()

        # Update coordinates
        self.coordinates.setText(f"X: {x}\nY: {y}\nZ: {z}")

class PoseEstimationApp(QMainWindow):
    def __init__(self, video_path_1, video_path_2, json_path):
        super().__init__()
        self.setWindowTitle("Pose Estimation and 3D Keypoint Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Variables for video paths
        self.video_path_1 = video_path_1
        self.video_path_2 = video_path_2
        self.json_path = json_path

        # Load JSON data
        self.json_data = load_json_data(json_path) if json_path else None

        # Start screen layout
        self.start_screen = QWidget()
        start_layout = QVBoxLayout()

        # Top-right calibrate button
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setFixedSize(100, 30)
        calibrate_layout = QHBoxLayout()
        calibrate_layout.addStretch()
        calibrate_layout.addWidget(self.calibrate_button)
        start_layout.addLayout(calibrate_layout)

        # Pose Estimation label
        pose_label = QLabel("Pose Estimation")
        pose_label.setAlignment(Qt.AlignCenter)
        pose_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        start_layout.addWidget(pose_label)

        # Live and Record buttons
        buttons_layout = QHBoxLayout()

        # Live button group
        live_button = QPushButton("Live")
        live_button.setFixedSize(100, 30)
        live_button_layout = QVBoxLayout()
        live_button_layout.addWidget(live_button)

        # Camera port buttons
        self.camera_buttons = []
        for i in range(4):
            camera_button = QPushButton(f"Camera {i}")
            camera_button.setFixedSize(100, 30)
            live_button_layout.addWidget(camera_button)
            self.camera_buttons.append(camera_button)

        buttons_layout.addLayout(live_button_layout)

        # Record button group
        record_button = QPushButton("Record")
        record_button.setFixedSize(100, 30)
        record_button_layout = QVBoxLayout()
        record_button_layout.addWidget(record_button)

        # File path inputs
        self.file_path_1 = QLineEdit()
        self.file_path_1.setPlaceholderText("Select file 1")
        self.file_path_1.setFixedSize(200, 30)
        self.file_path_1.setReadOnly(True)
        select_file_1 = QPushButton("...")
        select_file_1.setFixedSize(50, 30)
        select_file_1.clicked.connect(lambda: self.select_file(self.file_path_1))

        file_layout_1 = QHBoxLayout()
        file_layout_1.addWidget(self.file_path_1)
        file_layout_1.addWidget(select_file_1)

        self.file_path_2 = QLineEdit()
        self.file_path_2.setPlaceholderText("Select file 2")
        self.file_path_2.setFixedSize(200, 30)
        self.file_path_2.setReadOnly(True)
        select_file_2 = QPushButton("...")
        select_file_2.setFixedSize(50, 30)
        select_file_2.clicked.connect(lambda: self.select_file(self.file_path_2))

        file_layout_2 = QHBoxLayout()
        file_layout_2.addWidget(self.file_path_2)
        file_layout_2.addWidget(select_file_2)

        record_button_layout.addLayout(file_layout_1)
        record_button_layout.addLayout(file_layout_2)

        buttons_layout.addLayout(record_button_layout)

        start_layout.addLayout(buttons_layout)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.setFixedSize(200, 50)
        self.start_button.setStyleSheet("background-color: gray; color: white; font-size: 18px;")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_main_application)
        start_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        # Timer for enabling Start button
        self.enable_start_timer = QTimer()
        self.enable_start_timer.setSingleShot(True)
        self.enable_start_timer.timeout.connect(self.enable_start_button)

        # File path change detection
        self.file_path_1.textChanged.connect(self.check_start_conditions)
        self.file_path_2.textChanged.connect(self.check_start_conditions)

        self.start_screen.setLayout(start_layout)

        self.setCentralWidget(self.start_screen)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            line_edit.setText(file_path)

    def check_start_conditions(self):
        if self.file_path_1.text() and self.file_path_2.text():
            self.enable_start_timer.start(10000)  # 10 seconds delay

    def enable_start_button(self):
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 18px;")
        self.start_button.setEnabled(True)

    def start_main_application(self):
        video_path_1 = self.file_path_1.text()
        video_path_2 = self.file_path_2.text()
        json_path = self.json_path
        self.hide()
        self.pose_estimation_window = PoseEstimationMainApp(video_path_1, video_path_2, json_path)
        self.pose_estimation_window.show()

class PoseEstimationMainApp(QMainWindow):
    def __init__(self, video_path_1, video_path_2, json_path):
        super().__init__()
        self.setWindowTitle("Pose Estimation Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Video paths
        self.video_path_1 = video_path_1
        self.video_path_2 = video_path_2
        self.json_data = load_json_data(json_path)

        # Video capture objects
        self.cap1 = cv2.VideoCapture(self.video_path_1)
        self.cap2 = cv2.VideoCapture(self.video_path_2)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("Error: Unable to open videos.")
            sys.exit()

        # MediaPipe Pose models with high accuracy
        self.pose_model_1 = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose_model_2 = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Main screen
        self.main_screen = QWidget()

        # Layouts
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()
        button_layout = QVBoxLayout()

        # Video labels
        self.video_label_1 = QLabel("Video 1")
        self.video_label_2 = QLabel("Video 2")
        video_layout.addWidget(self.video_label_1)
        video_layout.addWidget(self.video_label_2)

        # 3D plot setup
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.canvas)

        # Buttons for keypoints
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        button_container = QWidget()
        self.button_layout = QVBoxLayout()
        button_container.setLayout(self.button_layout)
        self.scroll_area.setWidget(button_container)
        button_layout.addWidget(self.scroll_area)

        main_layout.addLayout(video_layout)
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(button_layout)
        self.main_screen.setLayout(main_layout)

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        # Connections for pose keypoints
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 26), (3, 4), (3, 11), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
            (7, 10), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (14, 17), (18, 19),
            (19, 20), (20, 21), (20, 32), (22, 23), (23, 24), (24, 25), (24, 33), (26, 27), (27, 28), (28, 29),
            (30, 31), (27, 30), (0, 22), (0, 18)
        ]

        self.frame_counter = 0
        self.json_frame_index = 0

        self.keypoint_buttons = []
        self.keypoint_dialogs = {}
        self.create_keypoint_buttons()

        self.stacked_layout = QVBoxLayout()
        self.stacked_layout.addWidget(self.main_screen)
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.stacked_layout)
        self.setCentralWidget(self.central_widget)

        self.start_processing()

    def create_keypoint_buttons(self):
        for i, name in enumerate(KEYPOINT_NAMES):
            button = QPushButton(name)
            button.clicked.connect(lambda checked, kp_index=i: self.open_or_focus_keypoint_dialog(kp_index))
            self.button_layout.addWidget(button)

    def open_or_focus_keypoint_dialog(self, kp_index):
        if kp_index not in self.keypoint_dialogs:
            dialog = KeypointDialog(KEYPOINT_NAMES[kp_index])
            self.keypoint_dialogs[kp_index] = dialog
            dialog.show()
        else:
            dialog = self.keypoint_dialogs[kp_index]
            dialog.activateWindow()

    def update_keypoint_dialogs(self):
        if self.json_data:
            timestamps = sorted(self.json_data.keys())
            if self.json_frame_index < len(timestamps):
                timestamp = timestamps[self.json_frame_index]
                body_list = self.json_data[timestamp]['body_list']
                if body_list:
                    keypoints = np.array(body_list[0]['keypoint'])
                    for kp_index, dialog in self.keypoint_dialogs.items():
                        if kp_index < len(keypoints):
                            x, y, z = keypoints[kp_index]
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                dialog.update_data(x, y, z)

    def start_processing(self):
        self.timer.start(300)  # Update every 300 ms (slower playback)

    def update_frames(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if not ret1 or not ret2 or self.json_frame_index >= len(self.json_data):
            self.timer.stop()
            self.cap1.release()
            self.cap2.release()
            return

        half_frame1 = frame1[:, :frame1.shape[1] // 2]
        half_frame2 = frame2[:, frame2.shape[1] // 2:]

        result1 = self.pose_model_1.process(cv2.cvtColor(half_frame1, cv2.COLOR_BGR2RGB))
        result2 = self.pose_model_2.process(cv2.cvtColor(half_frame2, cv2.COLOR_BGR2RGB))

        if result1.pose_landmarks:
            mp_drawing.draw_landmarks(half_frame1, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if result2.pose_landmarks:
            mp_drawing.draw_landmarks(half_frame2, result2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        self.display_frame(self.video_label_1, half_frame1)
        self.display_frame(self.video_label_2, half_frame2)

        self.update_3d_plot()
        self.update_keypoint_dialogs()

    def display_frame(self, label, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap)

    def update_3d_plot(self):
        self.canvas.axes.clear()
        if self.json_data:
            timestamps = sorted(self.json_data.keys())
            timestamp = timestamps[self.json_frame_index]
            body_list = self.json_data[timestamp]['body_list']
            for body in body_list:
                keypoints = np.array(body['keypoint'])
                if keypoints.size > 0:
                    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
                    if np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all():
                        self.canvas.axes.scatter(x, y, z, c='blue')
                        for conn in self.connections:
                            kp1, kp2 = conn
                            self.canvas.axes.plot([x[kp1], x[kp2]], [y[kp1], y[kp2]], [z[kp1], z[kp2]], color='red')
        self.canvas.axes.set_xlabel('X')
        self.canvas.axes.set_ylabel('Y')
        self.canvas.axes.set_zlabel('Z')
        self.canvas.draw()
        self.json_frame_index += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_path_1 = ""
    video_path_2 = ""
    json_path = ""

    window = PoseEstimationApp(video_path_1, video_path_2, json_path)
    window.show()

    sys.exit(app.exec_())

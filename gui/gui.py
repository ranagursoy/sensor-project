#!/usr/bin/env python3
"""
Real-time dual-camera pose viewer + 3-D key-point inspector
===========================================================

Modules
-------
• data_io      – load JSON meta-data
• mpl_canvas   – lightweight Matplotlib canvas wrapper
• dialogs      – on-demand 3-D key-point pop-ups
• video_thread – QThread that handles OpenCV capture & MediaPipe inference
• start_ui     – simple splash screen for choosing cameras / files
• main_ui      – side-by-side video playback + 3-D plot + key-point buttons

Run
---
$ python pose_viewer.py                # launches GUI
"""

from __future__ import annotations
import sys
import json
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple, Optional

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QMainWindow, QLineEdit, QFileDialog, QScrollArea, QTextEdit
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mediapipe as mp


# ------------------------------------------------------------------#
# 1.  Data I/O helpers
# ------------------------------------------------------------------#
def load_json(json_path: str) -> Dict:
    """Return parsed JSON (body-tracking meta-data)."""
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------#
# 2.  Tiny Matplotlib canvas
# ------------------------------------------------------------------#
class MplCanvas(FigureCanvas):
    """Embed a single 3-D Axes inside a Qt widget."""
    def __init__(self, w: float = 5, h: float = 4, dpi: int = 100):
        fig = Figure(figsize=(w, h), dpi=dpi)
        self.axes = fig.add_subplot(111, projection="3d")
        super().__init__(fig)


# ------------------------------------------------------------------#
# 3.  On-demand 3-D key-point dialog
# ------------------------------------------------------------------#
class KeypointDialog(QDialog):
    """
    Pop-up that shows (x, y, z) of one key-point and a tiny scatter plot.
    Keeps its own Matplotlib canvas so updates are cheap.
    """
    def __init__(self, kp_name: str):
        super().__init__()
        self.setWindowTitle(f"Key-point: {kp_name}")
        self.setFixedSize(400, 400)

        self.canvas = MplCanvas()
        self.txt = QTextEdit(readOnly=True)

        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        lay.addWidget(self.txt)

    # ---------- public API ----------------------------------------#
    def update_xyz(self, x: float, y: float, z: float) -> None:
        """Refresh scatter + text box with new coordinates."""
        ax = self.canvas.axes
        ax.clear()
        ax.scatter([x], [y], [z], c="blue")
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        self.canvas.draw()
        self.txt.setText(f"X: {x:.2f}  Y: {y:.2f}  Z: {z:.2f}")


# ------------------------------------------------------------------#
# 4.  Video / inference worker
# ------------------------------------------------------------------#
class VideoWorker(QThread):
    """
    Grabs frames from *two* videos (or cameras), runs MediaPipe Pose
    and emits numpy arrays for display.
    """
    frame_ready = pyqtSignal(np.ndarray, np.ndarray)            # half-frames
    keypoints_ready = pyqtSignal(np.ndarray)                    # (N, 33, 3)

    def __init__(self, src0: str, src1: str, parent=None):
        super().__init__(parent)
        self.src0, self.src1 = src0, src1
        self._stop = False

    # ---------- QThread interface --------------------------------#
    def run(self) -> None:
        cap0 = cv.VideoCapture(self.src0)
        cap1 = cv.VideoCapture(self.src1)
        if not cap0.isOpened() or not cap1.isOpened():
            print("[ERR] video/camera cannot be opened")
            return

        pose0 = mp.solutions.pose.Pose(model_complexity=2,
                                       min_detection_confidence=0.7,
                                       min_tracking_confidence=0.7)
        pose1 = mp.solutions.pose.Pose(model_complexity=2,
                                       min_detection_confidence=0.7,
                                       min_tracking_confidence=0.7)

        while not self._stop:
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            if not (ok0 and ok1):
                break

            half0 = f0[:, : f0.shape[1] // 2]
            half1 = f1[:, f1.shape[1] // 2:]

            res0 = pose0.process(cv.cvtColor(half0, cv.COLOR_BGR2RGB))
            res1 = pose1.process(cv.cvtColor(half1, cv.COLOR_BGR2RGB))

            # Draw landmarks for UI (doesn’t affect accuracy)
            if res0.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    half0, res0.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            if res1.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    half1, res1.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            stacked_kps = self._landmarks_to_xyz(res0)  # (33,3) or nan
            self.frame_ready.emit(half0, half1)
            self.keypoints_ready.emit(stacked_kps)

            self.msleep(30)  # ~33 fps

        cap0.release()
        cap1.release()

    # ---------- helpers ------------------------------------------#
    @staticmethod
    def _landmarks_to_xyz(result) -> np.ndarray:
        """Convert MediaPipe landmarks to (33,3) array or NaNs."""
        if not result.pose_landmarks:
            return np.full((33, 3), np.nan, dtype=np.float32)

        lm = result.pose_landmarks.landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    # ---------- public API ---------------------------------------#
    def stop(self) -> None:
        self._stop = True


# ------------------------------------------------------------------#
# 5.  Main window (viewer)
# ------------------------------------------------------------------#
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

CONNECTIONS = [
    (0, 11), (0, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)  # legs
]


class PoseViewer(QMainWindow):
    """High-level window that hosts two video panes + 3-D scatter plot."""
    def __init__(self, src0: str, src1: str, meta_json: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Dual-view Pose Viewer")
        self.setGeometry(80, 80, 1300, 800)

        # -- left panel: video streams --------------------------------------#
        self.lbl0 = QLabel(alignment=Qt.AlignCenter)
        self.lbl1 = QLabel(alignment=Qt.AlignCenter)
        vid_vbox = QVBoxLayout()
        vid_vbox.addWidget(self.lbl0)
        vid_vbox.addWidget(self.lbl1)
        vid_box = QWidget()
        vid_box.setLayout(vid_vbox)

        # -- center panel: 3-D Matplotlib plot ------------------------------#
        self.canvas = MplCanvas(5, 4, 100)

        # -- right panel: scroll-able key-point buttons ---------------------#
        btn_container = QWidget()
        btn_lay = QVBoxLayout(btn_container)

        self.btn_dialogs: Dict[int, KeypointDialog] = {}
        for kpi, name in enumerate(KEYPOINT_NAMES):
            pb = QPushButton(name)
            pb.clicked.connect(lambda _, i=kpi: self._toggle_dialog(i))
            btn_lay.addWidget(pb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(btn_container)

        # Assemble main h-box
        root_hbox = QHBoxLayout()
        root_hbox.addWidget(vid_box, 2)
        root_hbox.addWidget(self.canvas, 2)
        root_hbox.addWidget(scroll, 1)

        root = QWidget()
        root.setLayout(root_hbox)
        self.setCentralWidget(root)

        # -- background worker ---------------------------------------------#
        self.worker = VideoWorker(src0, src1)
        self.worker.frame_ready.connect(self._update_frames)
        self.worker.keypoints_ready.connect(self._update_plot_and_dialogs)
        self.worker.start()

        # -- optional JSON meta-data ---------------------------------------#
        self.meta = load_json(meta_json) if meta_json else None
        self.frame_idx = 0

    # ---------- slots -----------------------------------------------------#
    def _update_frames(self, f0: np.ndarray, f1: np.ndarray) -> None:
        """Convert BGR ndarray → QPixmap and show."""
        for frame, label in ((f0, self.lbl0), (f1, self.lbl1)):
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))

    def _update_plot_and_dialogs(self, kps: np.ndarray) -> None:
        """Refresh 3-D scatter + any open key-point dialogs."""
        self.canvas.axes.clear()
        if np.isfinite(kps).all():
            x, y, z = kps.T
            ax = self.canvas.axes
            ax.scatter(x, y, z, c="blue")
            for i, j in CONNECTIONS:
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c="red")

        self.canvas.draw()

        # Update open dialogs
        for idx, dlg in self.btn_dialogs.items():
            if np.isfinite(kps[idx]).all():
                dlg.update_xyz(*kps[idx])

        # If JSON was supplied, shift index for external data sync
        self.frame_idx += 1

    # ---------- helpers ---------------------------------------------------#
    def _toggle_dialog(self, kp_index: int) -> None:
        """Open or focus the 3-D window for a key-point."""
        dlg = self.btn_dialogs.get(kp_index)
        if dlg is None:
            dlg = KeypointDialog(KEYPOINT_NAMES[kp_index])
            dlg.show()
            self.btn_dialogs[kp_index] = dlg
        else:
            dlg.activateWindow()

    # ---------- Qt teardown ----------------------------------------------#
    def closeEvent(self, e) -> None:          # noqa: N802  (Qt override)
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(e)


# ------------------------------------------------------------------#
# 6.  Simple start/splash window
# ------------------------------------------------------------------#
class SplashScreen(QWidget):
    """
    Minimal splash: choose *two* files (or cameras) then launch viewer.
    The “Start” button activates after both file fields are populated for
    5 s – handy if cameras need warm-up.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Viewer – Launcher")

        # --- file chooser widgets ----------------------------------------#
        self.le0, self.le1 = QLineEdit(readOnly=True), QLineEdit(readOnly=True)
        self.b_choose0 = QPushButton("Browse…")
        self.b_choose1 = QPushButton("Browse…")
        self.b_start   = QPushButton("Start", enabled=False)

        self.b_choose0.clicked.connect(lambda: self._select_file(self.le0))
        self.b_choose1.clicked.connect(lambda: self._select_file(self.le1))
        self.b_start.clicked.connect(self._launch)

        # --- layout -------------------------------------------------------#
        row0 = QHBoxLayout(); row0.addWidget(self.le0); row0.addWidget(self.b_choose0)
        row1 = QHBoxLayout(); row1.addWidget(self.le1); row1.addWidget(self.b_choose1)

        vbox = QVBoxLayout(self)
        vbox.addLayout(row0)
        vbox.addLayout(row1)
        vbox.addWidget(self.b_start, alignment=Qt.AlignCenter)

        # --- timer gates start button after delay ------------------------#
        self.delay_timer = QTimer(singleShot=True, interval=5000)
        self.le0.textChanged.connect(self._maybe_enable_timer)
        self.le1.textChanged.connect(self._maybe_enable_timer)
        self.delay_timer.timeout.connect(self._enable_start)

    # ---------- internal --------------------------------------------------#
    def _select_file(self, line_edit: QLineEdit) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Select video / camera",
                                            "", "Video (*.mp4 *.avi)")
        if fn:
            line_edit.setText(fn)

    def _maybe_enable_timer(self) -> None:
        if self.le0.text() and self.le1.text():
            self.delay_timer.start()      # fire once after 5s

    def _enable_start(self) -> None:
        self.b_start.setEnabled(True)
        self.b_start.setStyleSheet("background:#2c7; color:white;")

    def _launch(self) -> None:
        self.viewer = PoseViewer(self.le0.text(), self.le1.text())
        self.viewer.show()
        self.close()


# ------------------------------------------------------------------#
# 7.  Entry-point
# ------------------------------------------------------------------#
def main() -> None:
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

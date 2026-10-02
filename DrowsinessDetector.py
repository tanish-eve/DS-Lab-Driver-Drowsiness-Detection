import queue
import threading
import time
import winsound
import cv2
import numpy as np
import sys
import os

from ultralytics import YOLO
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, pyqtSignal


class DrowsinessDetector(QMainWindow):
    # PyQt signal for thread-safe UI updates from worker thread
    frame_processed = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super().__init__()

        # --- Detection States & Counters ---
        self.yawn_state = 'No Yawn'
        self.left_eye_state = 'Open Eye'
        self.right_eye_state = 'Open Eye'
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0.0              # Current active eye closure duration in seconds
        self.microsleep_episodes = 0        # Count of completed microsleep events
        self.last_microsleep_duration = 0.0  # Duration of the most recent microsleep
        self.yawns = 0
        self.yawn_duration = 0.0            # Current active yawn duration in seconds
        self.last_yawn_duration = 0.0       # Duration of the most recent completed yawn

        # Internal timing & tracking flags
        self.eyes_closed = False
        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.last_alert_sound_time = 0.0

        # --- Detection Thresholds ---
        # EAR (Eye Aspect Ratio): Typical open eye is ~0.28-0.35, closed eye is ~0.15-0.23
        self.EAR_THRESHOLD = 0.25
        self.MICROSLEEP_TIME_THRESHOLD = 1.5   # Seconds of continuous eye closure for microsleep alert
        # MAR (Mouth Aspect Ratio): Typical closed/talking mouth is ~0.02-0.25, yawn is >0.50
        self.MAR_THRESHOLD = 0.50
        self.YAWN_ALERT_THRESHOLD = 4.0        # Seconds of continuous yawning for prolonged yawn alert

        # Landmark indices (MediaPipe Face Mesh 468 landmarks)
        # Right eye indices: [outer_corner, top1, top2, inner_corner, bot2, bot1]
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        # Left eye indices: [inner_corner, top1, top2, outer_corner, bot2, bot1]
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        # Mouth indices for MAR: top lip inner (13), bottom lip inner (14), corners (78, 308)
        self.MOUTH_TOP_BOT = (13, 14)
        self.MOUTH_CORNERS = (78, 308)

        # Legacy ROI landmark indices
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Load YOLO Models (with fallback protection) ---
        self.detectyawn = None
        self.detecteye = None
        yawn_model_path = "runs/detectyawn/train/weights/best.pt"
        eye_model_path = "runs/detecteye/train/weights/best.pt"

        if os.path.exists(yawn_model_path):
            try:
                self.detectyawn = YOLO(yawn_model_path)
            except Exception as e:
                print(f"Warning loading yawn model: {e}")

        if os.path.exists(eye_model_path):
            try:
                self.detecteye = YOLO(eye_model_path)
            except Exception as e:
                print(f"Warning loading eye model: {e}")

        # --- GUI Window Setup ---
        self.setWindowTitle("Real-Time Driver Drowsiness & Fatigue Detector")
        self.setGeometry(80, 80, 1020, 680)
        self.setStyleSheet("background-color: #121820; color: #FFFFFF;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # Left Container: Video Feed
        self.video_frame_container = QFrame(self)
        self.video_frame_container.setStyleSheet(
            "background-color: #1A2332; border: 2px solid #2A3B53; border-radius: 12px;"
        )
        video_layout = QVBoxLayout(self.video_frame_container)
        video_layout.setContentsMargins(10, 10, 10, 10)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border-radius: 8px; background-color: #0B0E14;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        self.main_layout.addWidget(self.video_frame_container)

        # Right Container: Telemetry & Alert Dashboard
        self.info_panel = QFrame(self)
        self.info_panel.setStyleSheet(
            "background-color: #1A2332; border: 2px solid #2A3B53; border-radius: 12px; padding: 15px;"
        )
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)

        self.info_label = QLabel(self)
        self.info_label.setStyleSheet("border: none; background: transparent;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        self.main_layout.addWidget(self.info_panel)

        self.update_info()

        # Connect thread-safe signal
        self.frame_processed.connect(self.on_frame_processed)

        # --- Camera & Threading ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.5)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)

        self.capture_thread.start()
        self.process_thread.start()

    # ---------------------------------------------------------
    # Core Mathematical Helper Functions (EAR & MAR)
    # ---------------------------------------------------------
    def calculate_ear(self, landmarks, eye_indices, iw, ih):
        """Calculates Eye Aspect Ratio (EAR) using 6 landmark points."""
        try:
            pts = [
                np.array([landmarks.landmark[i].x * iw, landmarks.landmark[i].y * ih])
                for i in eye_indices
            ]
            d_v1 = np.linalg.norm(pts[1] - pts[5])
            d_v2 = np.linalg.norm(pts[2] - pts[4])
            d_h = np.linalg.norm(pts[0] - pts[3])
            if d_h < 1e-6:
                return 0.0
            return (d_v1 + d_v2) / (2.0 * d_h)
        except Exception:
            return 0.0

    def calculate_mar(self, landmarks, iw, ih):
        """Calculates Mouth Aspect Ratio (MAR) using inner lip landmarks."""
        try:
            p_top = np.array([landmarks.landmark[self.MOUTH_TOP_BOT[0]].x * iw,
                              landmarks.landmark[self.MOUTH_TOP_BOT[0]].y * ih])
            p_bot = np.array([landmarks.landmark[self.MOUTH_TOP_BOT[1]].x * iw,
                              landmarks.landmark[self.MOUTH_TOP_BOT[1]].y * ih])
            p_left = np.array([landmarks.landmark[self.MOUTH_CORNERS[0]].x * iw,
                               landmarks.landmark[self.MOUTH_CORNERS[0]].y * ih])
            p_right = np.array([landmarks.landmark[self.MOUTH_CORNERS[1]].x * iw,
                                landmarks.landmark[self.MOUTH_CORNERS[1]].y * ih])
            d_h = np.linalg.norm(p_left - p_right)
            if d_h < 1e-6:
                return 0.0
            return np.linalg.norm(p_top - p_bot) / d_h
        except Exception:
            return 0.0

    # ---------------------------------------------------------
    # YOLO Model Predictions (Preserved & Protected)
    # ---------------------------------------------------------
    def predict_eye(self, eye_frame, eye_state):
        if self.detecteye is None or eye_frame is None or eye_frame.size == 0:
            return eye_state
        try:
            results_eye = self.detecteye.predict(eye_frame, verbose=False)
            boxes = results_eye[0].boxes
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])

            if class_id == 1:
                eye_state = "Close Eye"
            elif class_id == 0 and confidences[max_confidence_index] > 0.30:
                eye_state = "Open Eye"
        except Exception as e:
            pass
        return eye_state

    def predict_yawn(self, yawn_frame):
        if self.detectyawn is None or yawn_frame is None or yawn_frame.size == 0:
            return self.yawn_state
        try:
            results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
            boxes = results_yawn[0].boxes
            if len(boxes) == 0:
                return self.yawn_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])

            if class_id == 0 and confidences[max_confidence_index] > 0.60:
                self.yawn_state = "Yawn"
            elif class_id == 1 and confidences[max_confidence_index] > 0.50:
                self.yawn_state = "No Yawn"
        except Exception as e:
            pass
        return self.yawn_state

    # ---------------------------------------------------------
    # UI Dashboard Update
    # ---------------------------------------------------------
    def update_info(self, metrics=None):
        avg_ear = metrics.get('ear', 0.0) if metrics else 0.0
        mar = metrics.get('mar', 0.0) if metrics else 0.0

        # Status badge determination
        if self.alert_text:
            if "Microsleep" in self.alert_text:
                status_color = "#E53935"
                status_text = "⚠️ MICROSLEEP ALERT"
            else:
                status_color = "#FB8C00"
                status_text = "⚠️ PROLONGED YAWN"
        elif self.eyes_closed:
            status_color = "#FDD835"
            status_text = "🟡 EYE CLOSURE DETECTED"
        elif self.yawn_in_progress:
            status_color = "#FB8C00"
            status_text = "🟠 YAWN IN PROGRESS"
        else:
            status_color = "#43A047"
            status_text = "🟢 AWAKE & ALERT"

        # Alert HTML box
        alert_box_html = ""
        if self.alert_text:
            alert_box_html = (
                f"<div style='background-color: rgba(229, 57, 53, 0.2); border: 2px solid #E53935; "
                f"border-radius: 8px; padding: 10px; margin-bottom: 12px; text-align: center;'>"
                f"{self.alert_text}</div>"
            )

        info_text = (
            f"<div style='font-family: Segoe UI, sans-serif; color: #ECEFF1;'>"
            f"<h2 style='margin: 0 0 10px 0; color: #4FC3F7; font-size: 20px;'>Driver Telemetry</h2>"
            f"<div style='background-color: {status_color}; color: #FFFFFF; font-weight: bold; "
            f"padding: 8px 12px; border-radius: 6px; text-align: center; margin-bottom: 15px; font-size: 14px;'>"
            f"{status_text}</div>"
            f"{alert_box_html}"
            f"<table style='width: 100%; border-collapse: collapse; font-size: 14px;'>"
            f"<tr style='border-bottom: 1px solid #2A3B53;'>"
            f"<td style='padding: 8px 0;'><b>👁️ Blinks:</b></td>"
            f"<td style='text-align: right; color: #4FC3F7; font-weight: bold; font-size: 16px;'>{self.blinks}</td></tr>"
            f"<tr style='border-bottom: 1px solid #2A3B53;'>"
            f"<td style='padding: 8px 0;'><b>💤 Microsleep Events:</b></td>"
            f"<td style='text-align: right; color: #EF5350; font-weight: bold; font-size: 16px;'>{self.microsleep_episodes}</td></tr>"
            f"<tr style='border-bottom: 1px solid #2A3B53;'>"
            f"<td style='padding: 8px 0;'><b>⏱️ Eye Closure:</b></td>"
            f"<td style='text-align: right;'>{round(self.microsleeps, 2)}s <span style='color: #78909C; font-size: 11px;'>(last: {round(self.last_microsleep_duration, 2)}s)</span></td></tr>"
            f"<tr style='border-bottom: 1px solid #2A3B53;'>"
            f"<td style='padding: 8px 0;'><b>😮 Yawns:</b></td>"
            f"<td style='text-align: right; color: #FFA726; font-weight: bold; font-size: 16px;'>{self.yawns}</td></tr>"
            f"<tr style='border-bottom: 1px solid #2A3B53;'>"
            f"<td style='padding: 8px 0;'><b>⏳ Yawn Duration:</b></td>"
            f"<td style='text-align: right;'>{round(self.yawn_duration, 2)}s <span style='color: #78909C; font-size: 11px;'>(last: {round(self.last_yawn_duration, 2)}s)</span></td></tr>"
            f"</table>"
            f"<div style='margin-top: 15px; padding-top: 10px; border-top: 1px dashed #2A3B53; font-size: 12px; color: #90A4AE;'>"
            f"<div><b>EAR (Eye Aspect):</b> {avg_ear:.3f} <span style='color: #546E7A;'>(thresh: {self.EAR_THRESHOLD})</span></div>"
            f"<div style='margin-top: 4px;'><b>MAR (Mouth Aspect):</b> {mar:.3f} <span style='color: #546E7A;'>(thresh: {self.MAR_THRESHOLD})</span></div>"
            f"</div>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    # ---------------------------------------------------------
    # Frame Display & Alert Sound
    # ---------------------------------------------------------
    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def play_alert_sound(self):
        try:
            winsound.Beep(1200, 350)
        except Exception:
            pass

    def play_sound_in_thread(self):
        now = time.time()
        # Cooldown of 1.2s to prevent overlapping beeps
        if now - self.last_alert_sound_time > 1.2:
            self.last_alert_sound_time = now
            threading.Thread(target=self.play_alert_sound, daemon=True).start()

    # ---------------------------------------------------------
    # Background Threads: Capture & Processing
    # ---------------------------------------------------------
    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.01)

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            ih, iw, _ = frame.shape
            current_time = time.time()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            avg_ear = 0.0
            mar = 0.0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # 1. Compute EAR for both eyes
                right_ear = self.calculate_ear(face_landmarks, self.RIGHT_EYE_IDXS, iw, ih)
                left_ear = self.calculate_ear(face_landmarks, self.LEFT_EYE_IDXS, iw, ih)
                avg_ear = (right_ear + left_ear) / 2.0

                # 2. Compute MAR for mouth
                mar = self.calculate_mar(face_landmarks, iw, ih)

                # --- BLINK & MICROSLEEP LOGIC ---
                if avg_ear < self.EAR_THRESHOLD:
                    self.left_eye_state = "Close Eye"
                    self.right_eye_state = "Close Eye"

                    if not self.eyes_closed:
                        self.eyes_closed = True
                        self.eye_closed_start_time = current_time

                    self.microsleeps = current_time - self.eye_closed_start_time

                    # Check for prolonged microsleep
                    if self.microsleeps >= self.MICROSLEEP_TIME_THRESHOLD:
                        self.alert_text = "<span style='color: #EF5350; font-weight: bold;'>⚠️ Alert: Prolonged Microsleep Detected!</span>"
                        self.play_sound_in_thread()
                else:
                    self.left_eye_state = "Open Eye"
                    self.right_eye_state = "Open Eye"

                    if self.eyes_closed:
                        closure_duration = current_time - self.eye_closed_start_time
                        # Standard blink lasts between 0.06s and MICROSLEEP_TIME_THRESHOLD
                        if 0.06 <= closure_duration < self.MICROSLEEP_TIME_THRESHOLD:
                            self.blinks += 1
                        elif closure_duration >= self.MICROSLEEP_TIME_THRESHOLD:
                            self.microsleep_episodes += 1
                            self.last_microsleep_duration = closure_duration

                        self.eyes_closed = False
                        self.microsleeps = 0.0
                        if not self.yawn_in_progress:
                            self.alert_text = ''

                # --- YAWN LOGIC ---
                # Detect yawn through high MAR (mouth wide open)
                if mar >= self.MAR_THRESHOLD:
                    self.yawn_state = "Yawn"
                    if not self.yawn_in_progress:
                        self.yawn_in_progress = True
                        self.yawn_start_time = current_time

                    self.yawn_duration = current_time - self.yawn_start_time

                    # Check for prolonged yawn
                    if self.yawn_duration >= self.YAWN_ALERT_THRESHOLD:
                        self.alert_text = "<span style='color: #FFA726; font-weight: bold;'>⚠️ Alert: Prolonged Yawn Detected!</span>"
                        self.play_sound_in_thread()
                else:
                    self.yawn_state = "No Yawn"
                    if self.yawn_in_progress:
                        duration = current_time - self.yawn_start_time
                        if duration >= 1.0:  # Valid yawn must last at least 1 second
                            self.yawns += 1
                            self.last_yawn_duration = duration

                        self.yawn_in_progress = False
                        self.yawn_duration = 0.0
                        if not self.eyes_closed:
                            self.alert_text = ''

                # --- DRAW VISUAL HUD OVERLAY ON FRAME ---
                eye_color = (0, 0, 255) if self.eyes_closed else (0, 255, 0)
                mouth_color = (0, 140, 255) if self.yawn_in_progress else (0, 255, 0)

                # Draw landmark points for eyes
                for idx in self.RIGHT_EYE_IDXS + self.LEFT_EYE_IDXS:
                    pt = (int(face_landmarks.landmark[idx].x * iw), int(face_landmarks.landmark[idx].y * ih))
                    cv2.circle(frame, pt, 2, eye_color, -1)

                # Draw landmark points for mouth
                for idx in [self.MOUTH_TOP_BOT[0], self.MOUTH_TOP_BOT[1], self.MOUTH_CORNERS[0], self.MOUTH_CORNERS[1]]:
                    pt = (int(face_landmarks.landmark[idx].x * iw), int(face_landmarks.landmark[idx].y * ih))
                    cv2.circle(frame, pt, 3, mouth_color, -1)

                # HUD Top Banner
                cv2.rectangle(frame, (0, 0), (iw, 40), (20, 25, 35), -1)
                hud_text = f"EAR: {avg_ear:.2f} | MAR: {mar:.2f} | Blinks: {self.blinks} | Yawns: {self.yawns}"
                cv2.putText(frame, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # On-screen warning banner if alert is active
                if self.alert_text:
                    banner_text = "DROWSINESS ALERT!" if "Microsleep" in self.alert_text else "YAWN ALERT!"
                    banner_color = (0, 0, 255) if "Microsleep" in self.alert_text else (0, 140, 255)
                    cv2.rectangle(frame, (10, ih - 55), (iw - 10, ih - 10), banner_color, -1)
                    cv2.putText(frame, f"WARNING: {banner_text}", (20, ih - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # Emit signal to main GUI thread
            metrics = {'ear': avg_ear, 'mar': mar}
            self.frame_processed.emit(frame, metrics)

    def on_frame_processed(self, frame, metrics):
        """Executed on Qt Main GUI thread safely."""
        self.display_frame(frame)
        self.update_info(metrics)

    def closeEvent(self, event):
        """Clean shutdown of threads and camera on window close."""
        self.stop_event.set()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
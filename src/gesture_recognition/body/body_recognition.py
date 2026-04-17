"""
Body Pose Recognition API
============================
High-level API สำหรับ body pose recognition
รองรับ MAE + MLP modes, temporal smoothing, dataset import
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from typing import Union, List, Tuple, Dict, Optional

from gesture_recognition.body.body_mlp import BodyMLP
from gesture_recognition.core.temporal_smoother import TemporalSmoother
from gesture_recognition.core.utils import load_font, draw_thai_text, get_asset_path, load_dataset_from_folder


class BodyRecognition:

    def __init__(self, font_path: Optional[str] = None, font_size: int = 32,
                 mode: str = "mae", smoothing_window: int = 5):
        if font_path is None:
            font_path = get_asset_path()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.font = load_font(font_path, font_size)
        self.mode = mode
        self.body_mlp = BodyMLP()
        self.smoother = TemporalSmoother(window_size=smoothing_window) if smoothing_window > 0 else None

    def set_mode(self, mode: str):
        if mode in ("mae", "mlp"):
            self.mode = mode
            if self.smoother:
                self.smoother.reset()

    # ---- Model I/O ----

    def load_model(self, json_path: str) -> Dict:
        """Load a body pose model JSON saved by the Body GUI.

        Raises:
            FileNotFoundError: if the file does not exist. The message suggests
                opening the GUI to capture poses first.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Pose model not found: {json_path}\n"
                f"Tip: capture poses first with  python run_gui.py body  "
                f"(or  gesture-body  after pip install), then save the model "
                f"and point MODEL_PATH at that file."
            )
        with open(json_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
        if 'poses' not in model:
            model = {'poses': model}
        return model

    def train_mlp(self, model_data: Dict, epochs: int = 300, verbose: bool = True,
                  progress_callback=None) -> Optional[Dict]:
        self.body_mlp = BodyMLP()
        return self.body_mlp.train(model_data, epochs=epochs, verbose=verbose,
                                   progress_callback=progress_callback)

    def save_mlp(self, filepath: str):
        base = filepath.replace('.json', '').replace('.mlp', '')
        if self.body_mlp.is_trained:
            self.body_mlp.save(f"{base}_body.mlp.json")

    def load_mlp(self, filepath: str) -> bool:
        base = filepath.replace('.json', '').replace('.mlp', '')
        path = f"{base}_body.mlp.json"
        if os.path.exists(path):
            return self.body_mlp.load(path)
        return False

    # ---- Dataset loading ----

    def load_dataset(self, folder_path: str, progress_cb=None) -> Dict[str, List[list]]:
        """โหลด dataset จาก folder structure: folder/pose_name/img.jpg"""
        pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

        def detect_fn(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose_detector.process(rgb)
            if result.pose_landmarks:
                if BodyMLP.check_landmark_quality(result.pose_landmarks):
                    return result.pose_landmarks
            return None

        def extract_fn(landmarks):
            return BodyMLP.extract_features(landmarks)

        result = load_dataset_from_folder(folder_path, detect_fn, extract_fn, progress_cb)
        pose_detector.close()
        return result

    # ---- Prediction ----

    def predict_frame(self, model: Dict, frame: np.ndarray,
                      show_pose: int = 0) -> Tuple[str, float, np.ndarray]:
        """ทำนายท่าทางจาก frame, return (pose_name, confidence, output_image)"""
        output = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            if self.smoother:
                self.smoother.reset('body')
            return "", 0.0, output

        pose_lm = results.pose_landmarks
        if not BodyMLP.check_landmark_quality(pose_lm):
            return "", 0.0, output

        pose_name, confidence = self.recognize_from_landmarks(pose_lm, model)

        if self.smoother:
            self.smoother.update('body', pose_name, confidence)
            pose_name, confidence = self.smoother.get_smoothed('body')

        if show_pose == 1:
            output = self._draw_results(output, pose_name, confidence, results)

        return pose_name, confidence, output

    def recognize_from_landmarks(self, pose_landmarks, model: Dict) -> Tuple[str, float]:
        """จำแนกท่าทางจาก landmarks"""
        if self.mode == "mlp" and self.body_mlp.is_trained:
            return self.body_mlp.predict_from_landmarks(pose_landmarks)
        return self._recognize_mae(pose_landmarks, model)

    def _recognize_mae(self, pose_landmarks, model: Dict) -> Tuple[str, float]:
        """MAE recognition: weighted scoring (distances + angles*3)"""
        poses = model.get('poses', {})
        if not poses:
            return "", 0.0

        features = BodyMLP.extract_features(pose_landmarks)
        nd = BodyMLP.NUM_DISTANCES
        na = BodyMLP.NUM_ANGLES

        # Detect stored data format
        sample_dim = 0
        for stored_list in poses.values():
            if stored_list:
                sample_dim = len(stored_list[0])
                break
        has_angles = sample_dim >= nd + na

        scores = {}
        for name, stored_list in poses.items():
            if not stored_list:
                continue

            padded = []
            for stored in stored_list:
                s = np.array(stored)
                # Normalize stored distances by body size
                if len(s) >= nd:
                    dists = s[:nd].copy()
                    body_size = dists[BodyMLP.BODY_SIZE_INDEX] + 1e-8
                    dists = dists / body_size
                    s = np.concatenate([dists, s[nd:]]) if len(s) > nd else dists
                if len(s) < len(features):
                    s = np.concatenate([s, np.zeros(len(features) - len(s))])
                padded.append(s)

            centroid = np.mean(padded, axis=0)
            diff = np.abs(features - centroid)

            # Distance score (normalized)
            centroid_dist_mag = np.mean(np.abs(centroid[:nd])) + 1e-8
            score = np.mean(diff[:nd]) / centroid_dist_mag

            # Angle score (weighted x3)
            if has_angles and na > 0:
                centroid_angle_mag = np.mean(np.abs(centroid[nd:nd+na])) + 1e-8
                score += np.mean(diff[nd:nd+na]) / centroid_angle_mag * 3.0

            scores[name] = score

        if not scores:
            return "", 0.0

        best_match = min(scores, key=scores.get)
        best_score = scores[best_match]

        max_score = 4.0 if has_angles else 2.0
        if best_score > max_score:
            return "", 0.0

        if len(scores) >= 2:
            sorted_s = sorted(scores.values())
            margin = sorted_s[1] - best_score
            confidence = min(100, max(0, (margin / (sorted_s[1] + 1e-8)) * 150))
            if confidence > 10:
                return best_match, confidence
        else:
            if best_score < max_score * 0.5:
                confidence = min(100, max(0, (1 - best_score / max_score) * 100))
                return best_match, confidence

        return "", 0.0

    def _draw_results(self, image, pose_name, confidence, results):
        """วาด pose skeleton และชื่อท่าทางบนภาพ"""
        img = image.copy()
        h, w = img.shape[:2]

        # Draw skeleton
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Text overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if pose_name and confidence > 0:
            color = (0, 255, 0) if confidence >= 70 else (0, 165, 255)
            text = f"{pose_name} ({confidence:.1f}%)"
        else:
            color = (128, 128, 128)
            text = "ไม่รู้จัก"

        img = draw_thai_text(img, text, (20, 30), self.font, color)

        return img

    def close(self):
        """Release MediaPipe resources"""
        self.pose.close()

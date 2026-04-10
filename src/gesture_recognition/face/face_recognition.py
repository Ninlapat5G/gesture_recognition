"""
Face Expression Recognition API
=================================
High-level API สำหรับ face expression recognition
รองรับ MAE + MLP modes, temporal smoothing, dataset import
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from typing import Union, List, Tuple, Dict, Optional

from gesture_recognition.face.face_mlp import FaceMLP
from gesture_recognition.core.temporal_smoother import TemporalSmoother
from gesture_recognition.core.utils import load_font, draw_thai_text, get_asset_path, load_dataset_from_folder


class FaceRecognition:

    def __init__(self, font_path: Optional[str] = None, font_size: int = 32,
                 mode: str = "mae", smoothing_window: int = 5, use_3d: bool = True):
        if font_path is None:
            font_path = get_asset_path()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.font = load_font(font_path, font_size)
        self.mode = mode
        self.use_3d = use_3d
        self.face_mlp = FaceMLP(use_3d=use_3d)
        self.smoother = TemporalSmoother(window_size=smoothing_window) if smoothing_window > 0 else None

    def set_mode(self, mode: str):
        if mode in ("mae", "mlp"):
            self.mode = mode
            if self.smoother:
                self.smoother.reset()

    # ---- Model I/O ----

    def load_model(self, json_path: str) -> Dict:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
            if 'expressions' not in model:
                model = {'expressions': model}
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return {'expressions': {}}

    def train_mlp(self, model_data: Dict, epochs: int = 300, verbose: bool = True,
                  progress_callback=None) -> Optional[Dict]:
        self.face_mlp = FaceMLP(use_3d=self.use_3d)
        return self.face_mlp.train(model_data, epochs=epochs, verbose=verbose,
                                   progress_callback=progress_callback)

    def save_mlp(self, filepath: str):
        base = filepath.replace('.json', '').replace('.mlp', '')
        if self.face_mlp.is_trained:
            self.face_mlp.save(f"{base}_face.mlp.json")

    def load_mlp(self, filepath: str) -> bool:
        base = filepath.replace('.json', '').replace('.mlp', '')
        path = f"{base}_face.mlp.json"
        if os.path.exists(path):
            return self.face_mlp.load(path)
        return False

    # ---- Dataset loading ----

    def load_dataset(self, folder_path: str, progress_cb=None) -> Dict[str, List[list]]:
        face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )

        def detect_fn(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                if FaceMLP.check_landmark_quality(lm):
                    return lm
            return None

        def extract_fn(landmarks):
            return FaceMLP.extract_features(landmarks, self.use_3d)

        result = load_dataset_from_folder(folder_path, detect_fn, extract_fn, progress_cb)
        face_mesh.close()
        return result

    # ---- Prediction ----

    def predict_frame(self, model: Dict, frame: np.ndarray,
                      show_expression: int = 0) -> Tuple[str, float, np.ndarray]:
        output = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            if self.smoother:
                self.smoother.reset('face')
            return "", 0.0, output

        face_lm = results.multi_face_landmarks[0]
        if not FaceMLP.check_landmark_quality(face_lm):
            return "", 0.0, output

        expression, confidence = self.recognize_from_landmarks(face_lm, model)

        if self.smoother:
            self.smoother.update('face', expression, confidence)
            expression, confidence = self.smoother.get_smoothed('face')

        if show_expression == 1:
            output = self._draw_results(output, expression, confidence, results)

        return expression, confidence, output

    def recognize_from_landmarks(self, face_landmarks, model: Dict) -> Tuple[str, float]:
        if self.mode == "mlp" and self.face_mlp.is_trained:
            return self.face_mlp.predict_from_landmarks(face_landmarks)
        return self._recognize_mae(face_landmarks, model)

    def _recognize_mae(self, face_landmarks, model: Dict) -> Tuple[str, float]:
        """MAE with weighted scoring: distances + angles*3 + ratios*6"""
        expressions = model.get('expressions', {})
        if not expressions:
            return "", 0.0

        features = FaceMLP.extract_features(face_landmarks, self.use_3d)
        nd = FaceMLP.NUM_DISTANCES
        na = FaceMLP.NUM_ANGLES
        nr = FaceMLP.NUM_RATIOS

        # Detect stored data format
        sample_dim = 0
        for stored_list in expressions.values():
            if stored_list:
                sample_dim = len(stored_list[0])
                break
        has_angles = sample_dim >= nd + na
        has_ratios = sample_dim >= nd + na + nr

        scores = {}
        for name, stored_list in expressions.items():
            if not stored_list:
                continue

            padded = []
            for stored in stored_list:
                s = np.array(stored)
                if len(s) < len(features):
                    s = np.concatenate([s, np.zeros(len(features) - len(s))])
                padded.append(s)

            centroid = np.mean(padded, axis=0)
            diff = np.abs(features - centroid)

            centroid_dist_mag = np.mean(np.abs(centroid[:nd])) + 1e-8
            score = np.mean(diff[:nd]) / centroid_dist_mag

            if has_angles and na > 0:
                centroid_angle_mag = np.mean(np.abs(centroid[nd:nd+na])) + 1e-8
                score += np.mean(diff[nd:nd+na]) / centroid_angle_mag * 3.0

            if has_ratios and nr > 0:
                centroid_ratio_mag = np.mean(np.abs(centroid[nd+na:nd+na+nr])) + 1e-8
                score += np.mean(diff[nd+na:nd+na+nr]) / centroid_ratio_mag * 6.0

            scores[name] = score

        if not scores:
            return "", 0.0

        best_match = min(scores, key=scores.get)
        best_score = scores[best_match]

        max_score = 10.0 if (has_angles and has_ratios) else 2.0
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

    def _draw_results(self, image, expression, confidence, results):
        img = image.copy()
        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if expression and confidence > 0:
            color = (0, 255, 0) if confidence >= 70 else (0, 165, 255)
            text = f"{expression} ({confidence:.1f}%)"
        else:
            color = (128, 128, 128)
            text = "ไม่รู้จัก"

        img = draw_thai_text(img, text, (20, 30), self.font, color)

        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, lm,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        return img

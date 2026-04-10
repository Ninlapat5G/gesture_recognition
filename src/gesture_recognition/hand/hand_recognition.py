"""
Hand Gesture Recognition API
==============================
High-level API สำหรับ hand gesture recognition
รองรับ MAE + MLP modes, single + both hands, temporal smoothing
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from PIL import ImageFont
from typing import Union, List, Tuple, Dict, Optional

from gesture_recognition.hand.hand_mlp import HandMLP
from gesture_recognition.core.temporal_smoother import TemporalSmoother
from gesture_recognition.core.utils import load_font, draw_thai_text, get_asset_path, load_dataset_from_folder


class HandRecognition:
    """High-level API สำหรับ hand gesture recognition"""

    def __init__(self, font_path: Optional[str] = None, font_size: int = 32,
                 mode: str = "mae", smoothing_window: int = 5):
        if font_path is None:
            font_path = get_asset_path()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.font = load_font(font_path, font_size)
        self.font_size = font_size

        # Position tracking state
        self.initial_position = None
        self.current_position = [0, 0]

        # Recognition mode
        self.mode = mode
        self.mlp_left = HandMLP()
        self.mlp_right = HandMLP()
        self.mlp_both = HandMLP(input_size=861)

        # Temporal smoother
        self.smoother = TemporalSmoother(window_size=smoothing_window) if smoothing_window > 0 else None

    def set_mode(self, mode: str):
        if mode not in ("mae", "mlp"):
            return
        self.mode = mode
        if self.smoother:
            self.smoother.reset()

    # ---- Model I/O ----

    def load_model(self, json_path: str) -> Dict:
        """โหลด gesture model จาก JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                model = json.load(f)
            if 'left' not in model or 'right' not in model:
                raise ValueError("โมเดลต้องมี 'left' และ 'right'")
            if 'both' not in model:
                model['both'] = {}
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return {'left': {}, 'right': {}, 'both': {}}

    def train_mlp(self, model: Dict, epochs: int = 200, verbose: bool = True,
                  progress_callback=None) -> Dict:
        """Train MLP สำหรับทุก hand type ที่มีข้อมูล >= 2 classes"""
        results = {}
        hand_names = {'left': 'มือซ้าย', 'right': 'มือขวา', 'both': '2 มือ'}

        for hand_type, mlp in [('left', self.mlp_left), ('right', self.mlp_right),
                                ('both', self.mlp_both)]:
            gestures = model.get(hand_type, {})
            if len(gestures) >= 2:
                if verbose:
                    print(f"\nTraining MLP {hand_names[hand_type]}...")

                cb = None
                if progress_callback:
                    cb = lambda ep, tl, ta, vl, va, ht=hand_type: progress_callback(ht, ep, tl, ta, vl, va)

                stats = mlp.train(model, hand_type, epochs=epochs,
                                  verbose=verbose, progress_callback=cb)
                results[hand_type] = stats
            else:
                results[hand_type] = None

        return results

    def save_mlp(self, filepath: str):
        """บันทึก trained MLP models"""
        base = filepath.replace('.json', '').replace('.mlp', '')
        for suffix, mlp in [('_left.mlp.json', self.mlp_left),
                             ('_right.mlp.json', self.mlp_right),
                             ('_both.mlp.json', self.mlp_both)]:
            if mlp.is_trained:
                mlp.save(f"{base}{suffix}")

    def load_mlp(self, filepath: str) -> bool:
        """โหลด trained MLP models"""
        base = filepath.replace('.json', '').replace('.mlp', '')
        loaded = False
        for suffix, mlp, name in [('_left.mlp.json', self.mlp_left, 'มือซ้าย'),
                                   ('_right.mlp.json', self.mlp_right, 'มือขวา'),
                                   ('_both.mlp.json', self.mlp_both, '2 มือ')]:
            path = f"{base}{suffix}"
            if os.path.exists(path) and mlp.load(path):
                loaded = True
        return loaded

    # ---- Dataset loading ----

    def load_dataset(self, folder_path: str, hand_type: str = 'right',
                     progress_cb=None) -> Dict[str, List[list]]:
        """
        โหลด dataset จาก folder (subfolder per gesture)
        detect มือจากรูป แล้ว extract 225-dim features

        Returns: {gesture_name: [[features], ...]}
        """
        hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        def detect_fn(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)
            if result.multi_hand_landmarks:
                return result.multi_hand_landmarks[0]
            return None

        def extract_fn(landmarks):
            return HandMLP.extract_single_hand_features(landmarks)

        result = load_dataset_from_folder(folder_path, detect_fn, extract_fn, progress_cb)
        hands_detector.close()
        return result

    # ---- Prediction ----

    def predict_frame(self, model: Dict, frame: np.ndarray,
                      show_gesture: int = 0) -> Tuple[str, List[str], List[float], np.ndarray]:
        """
        ทำนายท่าทางมือจาก frame

        Returns: (status, gesture_list, confidence_list, output_frame)
        """
        output_frame = frame.copy()
        status, gestures, confidences, results = self._predict_with_results(model, frame)

        if show_gesture == 1:
            output_frame = self._draw_results(output_frame, status, gestures, confidences, results)

        return status, gestures, confidences, output_frame

    def predict_frame_with_position(self, model: Dict, frame: np.ndarray,
                                     confidence_threshold: float = 70.0,
                                     landmark_index: int = 9,
                                     show_gesture: int = 0,
                                     scale: float = 100.0
                                     ) -> Tuple[str, List[str], List[float], List[float], np.ndarray]:
        """
        ทำนายท่าทางมือพร้อม position tracking

        Args:
            landmark_index: 0=WRIST, 4=THUMB_TIP, 8=INDEX_TIP, 9=MID_MCP, 12=MID_TIP

        Returns: (status, gestures, confidences, position, output_frame)
        """
        output_frame = frame.copy()
        status, gestures, confidences, results = self._predict_with_results(model, frame)

        # Position tracking
        if results and results.multi_hand_landmarks and confidences:
            max_idx = confidences.index(max(confidences))
            if confidences[max_idx] >= confidence_threshold:
                lm = results.multi_hand_landmarks[max_idx]
                if landmark_index < len(lm.landmark):
                    pt = lm.landmark[landmark_index]
                    current_pos = [pt.x, pt.y]

                    if self.initial_position is None:
                        self.initial_position = current_pos
                        self.current_position = [0, 0]
                    else:
                        self.current_position = [
                            (current_pos[0] - self.initial_position[0]) * scale,
                            -(current_pos[1] - self.initial_position[1]) * scale
                        ]
                else:
                    self._reset_position()
            else:
                self._reset_position()
        else:
            self._reset_position()

        if show_gesture == 1:
            output_frame = self._draw_results(output_frame, status, gestures, confidences,
                                               results, self.current_position, landmark_index)

        return status, gestures, confidences, self.current_position.copy(), output_frame

    def predict_image(self, model: Dict, image_input: Union[str, np.ndarray],
                      show_gesture: int = 0) -> Tuple[str, List[str], List[float]]:
        """ทำนายจากรูปภาพ"""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return "ไม่พบ", [], []
        else:
            img = image_input.copy()

        status, gestures, confidences, results = self._predict_with_results(model, img)

        if show_gesture == 1:
            img = self._draw_results(img, status, gestures, confidences, results)
            cv2.imshow("Gesture Recognition", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return status, gestures, confidences

    # ---- Recognition methods that accept landmarks (for GUI integration) ----

    def recognize_from_landmarks(self, landmarks, hand_type: str,
                                  model: Dict) -> Tuple[str, float]:
        """จับคู่ท่าทางจาก landmarks (MAE หรือ MLP ตาม self.mode)"""
        if self.mode == "mlp":
            mlp = self.mlp_left if hand_type == 'left' else self.mlp_right
            if not mlp.is_trained:
                return "", 0.0
            return mlp.predict_from_landmarks(landmarks)
        else:
            return self._recognize_mae(landmarks, hand_type, model)

    def recognize_both_from_landmarks(self, left_landmarks, right_landmarks,
                                       model: Dict) -> Tuple[str, float]:
        """จับคู่ท่าทาง 2 มือจาก landmarks"""
        if self.mode == "mlp":
            if not self.mlp_both.is_trained:
                return "", 0.0
            return self.mlp_both.predict_from_both_landmarks(left_landmarks, right_landmarks)
        else:
            return self._recognize_mae_both(left_landmarks, right_landmarks, model)

    # ---- Internal ----

    def _predict_with_results(self, model: Dict, image: np.ndarray):
        """ทำนายพร้อมส่ง MediaPipe results กลับ (ไม่ต้อง process ซ้ำ)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            if self.smoother:
                self.smoother.reset()
            return "ไม่พบ", [], [], results

        left_result = None
        right_result = None
        left_landmarks = None
        right_landmarks = None

        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            hand_type = 'left' if hand_label == 'Left' else 'right'

            gesture, confidence = self.recognize_from_landmarks(landmarks, hand_type, model)

            if self.smoother:
                self.smoother.update(hand_type, gesture, confidence)
                gesture, confidence = self.smoother.get_smoothed(hand_type)

            if hand_type == 'left':
                left_result = (gesture, confidence)
                left_landmarks = landmarks
            else:
                right_result = (gesture, confidence)
                right_landmarks = landmarks

        gestures = []
        confidences = []

        if left_result:
            gestures.append(left_result[0])
            confidences.append(left_result[1])
        if right_result:
            gestures.append(right_result[0])
            confidences.append(right_result[1])

        # Both-hands recognition
        if left_landmarks and right_landmarks and model.get('both'):
            both_gesture, both_conf = self.recognize_both_from_landmarks(
                left_landmarks, right_landmarks, model
            )
            if both_conf > 0:
                if self.smoother:
                    self.smoother.update('both', both_gesture, both_conf)
                    both_gesture, both_conf = self.smoother.get_smoothed('both')
                gestures.append(both_gesture)
                confidences.append(both_conf)

        if left_result and right_result:
            status = "ทั้งหมด"
        elif left_result:
            status = "ซ้าย"
        elif right_result:
            status = "ขวา"
        else:
            status = "ไม่พบ"

        return status, gestures, confidences, results

    def _recognize_mae(self, landmarks, hand_type: str, model: Dict) -> Tuple[str, float]:
        """MAE matching with hand-size normalization"""
        distances = HandMLP.calculate_distances(landmarks)
        distances = HandMLP.normalize_distances_by_hand_size(distances)

        best_match = None
        best_score = float('inf')

        for name, stored_list in model.get(hand_type, {}).items():
            for stored in stored_list:
                stored_arr = np.array(stored[:210])
                # Normalize stored distances too
                stored_size = stored_arr[HandMLP.HAND_SIZE_INDEX] + 1e-8
                stored_norm = stored_arr / stored_size
                score = np.mean(np.abs(distances - stored_norm))
                if score < best_score:
                    best_score = score
                    best_match = name

        if best_score < 0.1:
            confidence = 100 * (1 - best_score / 0.1)
            return best_match, confidence
        return "", 0.0

    def _recognize_mae_both(self, left_lm, right_lm, model: Dict) -> Tuple[str, float]:
        """MAE matching สำหรับ 2 มือ"""
        features = HandMLP.extract_both_hands_features(left_lm, right_lm)

        best_match = None
        best_score = float('inf')

        for name, stored_list in model.get('both', {}).items():
            for stored in stored_list:
                score = np.mean(np.abs(features - np.array(stored)))
                if score < best_score:
                    best_score = score
                    best_match = name

        if best_score < 0.15:
            confidence = 100 * (1 - best_score / 0.15)
            return best_match, confidence
        return "", 0.0

    def _draw_results(self, image, status, gestures, confidences, results,
                      position=None, landmark_index=None):
        """วาดผลทำนายบนภาพ — ใช้ results ที่ส่งมา ไม่ process ซ้ำ"""
        img = image.copy()
        h, w = img.shape[:2]

        overlay = img.copy()
        box_h = 200 if position else 150
        cv2.rectangle(overlay, (10, 10), (w - 10, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        status_colors = {
            "ไม่พบ": (0, 0, 255),
            "ซ้าย": (0, 165, 255),
            "ขวา": (0, 165, 255),
            "ทั้งหมด": (0, 255, 0)
        }
        color = status_colors.get(status, (255, 255, 255))
        img = draw_thai_text(img, f"Status: {status}", (20, 20), self.font, color)

        y_offset = 60
        for i, (gesture, conf) in enumerate(zip(gestures, confidences)):
            label = "Left" if i == 0 and status in ["ซ้าย", "ทั้งหมด"] else "Right"
            text = f"{label}: {gesture} ({conf:.1f}%)"
            img = draw_thai_text(img, text, (20, y_offset), self.font, (0, 255, 0))
            y_offset += 40

        if position is not None:
            pos_text = f"Position: X={position[0]:.1f}, Y={position[1]:.1f}"
            if landmark_index is not None:
                pos_text = f"Position (L{landmark_index}): X={position[0]:.1f}, Y={position[1]:.1f}"
            pos_color = (0, 255, 255) if position != [0, 0] else (128, 128, 128)
            img = draw_thai_text(img, pos_text, (20, y_offset), self.font, pos_color)

        # Draw skeleton ใช้ results ที่ส่งมา (ไม่ process ซ้ำ)
        if results and results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
        return img

    def _reset_position(self):
        self.initial_position = None
        self.current_position = [0, 0]

    # ---- Utility methods ----

    def compare_models(self, models_list: List[Dict],
                       images_list: List[Union[str, np.ndarray]],
                       ground_truth_list: List[str],
                       model_names: List[str] = None) -> Optional[Dict]:
        """เปรียบเทียบความแม่นยำของหลายโมเดล"""
        if len(images_list) != len(ground_truth_list):
            return None

        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models_list))]

        correct_counts = {name: 0 for name in model_names}
        predictions = {name: [] for name in model_names}

        for img_idx, (image, gt) in enumerate(zip(images_list, ground_truth_list)):
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    continue
            else:
                img = image

            for model, name in zip(models_list, model_names):
                status, gestures, confidences, _ = self._predict_with_results(model, img)
                predicted = gestures[0] if gestures else "ไม่พบ"
                is_correct = predicted == gt
                if is_correct:
                    correct_counts[name] += 1
                predictions[name].append({
                    'predicted': predicted,
                    'confidence': confidences[0] if confidences else 0,
                    'correct': is_correct
                })

        total = len(images_list)
        accuracy = {n: correct_counts[n] / total * 100 for n in model_names}
        winner = max(accuracy, key=accuracy.get)

        return {
            'results': predictions,
            'accuracy': accuracy,
            'correct_counts': correct_counts,
            'total_images': total,
            'winner': winner
        }

    @staticmethod
    def merge_models(models_list: List[Dict]) -> Dict:
        """รวมหลายโมเดลเข้าด้วยกัน"""
        merged = {'left': {}, 'right': {}, 'both': {}}
        for hand_type in ['left', 'right', 'both']:
            for model in models_list:
                for name, variations in model.get(hand_type, {}).items():
                    if name not in merged[hand_type]:
                        merged[hand_type][name] = []
                    merged[hand_type][name].extend(variations)
        return merged

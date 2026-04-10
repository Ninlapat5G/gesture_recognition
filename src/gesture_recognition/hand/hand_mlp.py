"""
Hand Gesture MLP Classifier
=============================
MLP classifier สำหรับจำแนกท่าทางมือ
รองรับมือเดี่ยว (225 dims: 210 distances + 15 angles) และ 2 มือ (861 dims)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from gesture_recognition.core.base_mlp import BaseMLP


class HandMLP(BaseMLP):
    """
    MLP Classifier สำหรับจำแนกท่าทางมือ

    Features:
    - มือเดี่ยว: 210 pairwise distances + 15 joint angles = 225 dims
    - สองมือ: 210 left + 210 right + 441 cross-hand = 861 dims
    """

    ANGLE_TRIPLETS = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8),        # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12),   # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20), # Pinky
    ]

    HAND_SIZE_INDEX = 8  # pair(0,9) = wrist to middle_finger_MCP

    def __init__(self, input_size: int = 225, hidden_sizes: List[int] = None,
                 learning_rate: float = 0.001):
        if hidden_sizes is None:
            if input_size > 500:  # both-hands (861)
                hidden_sizes = [256, 128]
            else:
                hidden_sizes = [128, 64]
        super().__init__(input_size, hidden_sizes, learning_rate)

    # ---- Feature extraction (single hand) ----

    @staticmethod
    def calculate_distances(landmarks) -> np.ndarray:
        """Pairwise distances ระหว่าง 21 landmarks → 210 features (vectorized)"""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        diff = pts[:, None, :] - pts[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
        iu = np.triu_indices(len(pts), k=1)
        return dist_matrix[iu]

    @staticmethod
    def calculate_angles(landmarks) -> np.ndarray:
        """Joint angles จาก landmarks → 15 features (normalized to [0,1])"""
        angles = []
        for a_idx, b_idx, c_idx in HandMLP.ANGLE_TRIPLETS:
            a = landmarks.landmark[a_idx]
            b = landmarks.landmark[b_idx]
            c = landmarks.landmark[c_idx]

            ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
            bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) / np.pi
            angles.append(angle)

        return np.array(angles)

    @staticmethod
    def normalize_distances_by_hand_size(distances: np.ndarray) -> np.ndarray:
        """Normalize distances ด้วย hand_size"""
        hand_size = distances[HandMLP.HAND_SIZE_INDEX] + 1e-8
        return distances / hand_size

    @staticmethod
    def extract_single_hand_features(landmarks) -> np.ndarray:
        """Extract full 225-dim feature vector จาก landmarks (distances + real angles)"""
        distances = HandMLP.calculate_distances(landmarks)
        distances = HandMLP.normalize_distances_by_hand_size(distances)
        angles = HandMLP.calculate_angles(landmarks)
        return np.concatenate([distances, angles])

    # ---- Feature extraction (both hands) ----

    @staticmethod
    def calculate_cross_hand_distances(left_landmarks, right_landmarks) -> np.ndarray:
        """Distances ระหว่าง landmarks มือซ้าย/ขวา → 441 features (21x21, vectorized)"""
        left_pts = np.array([[lm.x, lm.y, lm.z] for lm in left_landmarks.landmark])
        right_pts = np.array([[lm.x, lm.y, lm.z] for lm in right_landmarks.landmark])
        diff = left_pts[:, None, :] - right_pts[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1)).ravel()

    @staticmethod
    def extract_both_hands_features(left_landmarks, right_landmarks) -> np.ndarray:
        """Feature vector สำหรับ 2 มือ → 861 features"""
        left_dists = HandMLP.calculate_distances(left_landmarks)
        left_hand_size = left_dists[HandMLP.HAND_SIZE_INDEX] + 1e-8
        left_norm = left_dists / left_hand_size

        right_dists = HandMLP.calculate_distances(right_landmarks)
        right_hand_size = right_dists[HandMLP.HAND_SIZE_INDEX] + 1e-8
        right_norm = right_dists / right_hand_size

        cross_dists = HandMLP.calculate_cross_hand_distances(left_landmarks, right_landmarks)
        avg_hand_size = (left_hand_size + right_hand_size) / 2
        cross_norm = cross_dists / avg_hand_size

        return np.concatenate([left_norm, right_norm, cross_norm])

    # ---- Training ----

    def train(self, model_data: Dict, hand_type: str, epochs: int = 300,
              batch_size: int = 32, verbose: bool = True,
              progress_callback=None) -> Optional[Dict]:
        """Train MLP จาก gesture model JSON"""
        gestures = model_data.get(hand_type, {})

        if len(gestures) < 2:
            if verbose:
                print(f"ต้องมีอย่างน้อย 2 ท่าทาง แต่มี {len(gestures)} ท่าทาง")
            return None

        X_list = []
        y_list = []
        self.classes = sorted(gestures.keys())
        class_to_idx = {name: i for i, name in enumerate(self.classes)}

        for name, samples in gestures.items():
            for sample in samples:
                if hand_type == 'both':
                    if len(sample) == 861:
                        feature = np.array(sample)
                    else:
                        continue
                else:
                    if len(sample) == 225:
                        # New format: 210 distances + 15 real angles
                        dists = np.array(sample[:210])
                        angles = np.array(sample[210:225])
                        hand_size = dists[self.HAND_SIZE_INDEX] + 1e-8
                        dists = dists / hand_size
                        feature = np.concatenate([dists, angles])
                    elif len(sample) == 210:
                        # Old format: distances only, pad zeros for angles
                        dists = np.array(sample)
                        hand_size = dists[self.HAND_SIZE_INDEX] + 1e-8
                        dists = dists / hand_size
                        feature = np.concatenate([dists, np.zeros(15)])
                    else:
                        continue

                X_list.append(feature)
                y_list.append(class_to_idx[name])

        # Wrap progress_callback to match BaseMLP signature
        cb = None
        if progress_callback:
            cb = lambda epoch, tl, ta, vl, va: progress_callback(epoch, tl, ta, vl, va)

        return self.train_from_data(
            X_list, y_list, self.classes,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, progress_callback=cb
        )

    # ---- Prediction ----

    def predict_from_landmarks(self, landmarks) -> Tuple[str, float]:
        """ทำนายท่ามือเดี่ยวจาก MediaPipe landmarks"""
        # Auto-detect: model trained with real angles or zeros?
        use_real_angles = True
        if self.feature_mean is not None and len(self.feature_mean) >= 225:
            angles_mean = np.mean(np.abs(self.feature_mean[210:225]))
            if angles_mean < 0.001:
                use_real_angles = False

        distances = self.calculate_distances(landmarks)
        distances = self.normalize_distances_by_hand_size(distances)

        if use_real_angles:
            angles = self.calculate_angles(landmarks)
        else:
            angles = np.zeros(15)

        features = np.concatenate([distances, angles])
        return self.predict(features)

    def predict_from_both_landmarks(self, left_landmarks, right_landmarks) -> Tuple[str, float]:
        """ทำนายท่า 2 มือ"""
        features = self.extract_both_hands_features(left_landmarks, right_landmarks)
        return self.predict(features)


# Backward compat alias
GestureMLP = HandMLP

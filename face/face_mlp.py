"""
Face Expression MLP Classifier
================================
MLP classifier สำหรับจำแนกท่าทางใบหน้า
ใช้ 50 key landmarks จาก MediaPipe Face Mesh → 1265 features
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional

from core.base_mlp import BaseMLP


class FaceMLP(BaseMLP):
    """
    MLP Classifier สำหรับจำแนกท่าทางใบหน้า

    Features:
    - 1225 pairwise distances (normalized by face size)
    - 25 facial angles (rotation-invariant)
    - 15 expression-specific ratios (head-pose-invariant)
    - Total: 1265 dims

    Architecture: Input(1265) → Hidden(256, ReLU) → Hidden(128, ReLU) → Output(softmax)
    """

    KEY_LANDMARKS = {
        'left_eye':      [33, 160, 158, 133, 153, 144],
        'right_eye':     [362, 385, 387, 263, 373, 380],
        'left_eyebrow':  [46, 53, 52, 65, 55],
        'right_eyebrow': [276, 283, 282, 295, 285],
        'nose':          [1, 168, 6, 197, 195],
        'mouth_outer':   [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375],
        'mouth_inner':   [78, 191, 80, 81, 82, 13, 312, 311],
        'jaw_chin':      [10, 152, 234],
    }

    ALL_INDICES: List[int] = []
    for _group in KEY_LANDMARKS.values():
        ALL_INDICES.extend(_group)
    NUM_LANDMARKS = len(ALL_INDICES)  # 50
    NUM_DISTANCES = NUM_LANDMARKS * (NUM_LANDMARKS - 1) // 2  # 1225

    FACE_SIZE_LANDMARKS = (33, 263)

    FACE_ANGLE_TRIPLETS = [
        (160, 33, 144), (158, 133, 153), (160, 144, 133),
        (385, 362, 380), (387, 263, 373), (385, 380, 263),
        (46, 53, 55), (52, 65, 55),
        (276, 283, 285), (282, 295, 285),
        (61, 0, 291), (61, 13, 291), (40, 0, 270), (80, 13, 312),
        (185, 0, 409), (37, 0, 267),
        (168, 6, 195), (1, 6, 197), (168, 1, 195),
        (234, 152, 10), (33, 152, 263), (61, 152, 291),
        (33, 1, 263), (46, 6, 276), (0, 13, 152),
    ]
    NUM_ANGLES = len(FACE_ANGLE_TRIPLETS)  # 25

    RATIO_PAIRS = [
        ((160, 144), (33, 133)), ((158, 153), (33, 133)),
        ((385, 380), (362, 263)), ((387, 373), (362, 263)),
        ((13, 0), (61, 291)), ((80, 82), (61, 291)), ((81, 13), (61, 291)),
        ((53, 160), (33, 133)), ((283, 385), (362, 263)),
        ((52, 158), (33, 133)), ((282, 387), (362, 263)),
        ((61, 1), (0, 152)), ((291, 1), (0, 152)),
        ((1, 0), (1, 152)), ((6, 13), (33, 263)),
    ]
    NUM_RATIOS = len(RATIO_PAIRS)  # 15
    INPUT_SIZE = NUM_DISTANCES + NUM_ANGLES + NUM_RATIOS  # 1265

    def __init__(self, hidden_sizes: List[int] = None, learning_rate: float = 0.001,
                 use_3d: bool = True):
        super().__init__(
            input_size=self.INPUT_SIZE,
            hidden_sizes=hidden_sizes or [256, 128],
            learning_rate=learning_rate
        )
        self.use_3d = use_3d

    @staticmethod
    def get_face_size(face_landmarks, use_3d: bool = True) -> float:
        lm = face_landmarks.landmark
        a, b = lm[33], lm[263]
        if use_3d:
            return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) + 1e-8
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2) + 1e-8

    @staticmethod
    def calculate_distances(face_landmarks, use_3d: bool = True) -> np.ndarray:
        """Pairwise distances across 50 key landmarks → 1225 features (vectorized)"""
        lm = face_landmarks.landmark
        if use_3d:
            pts = np.array([[lm[idx].x, lm[idx].y, lm[idx].z] for idx in FaceMLP.ALL_INDICES])
        else:
            pts = np.array([[lm[idx].x, lm[idx].y] for idx in FaceMLP.ALL_INDICES])
        diff = pts[:, None, :] - pts[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
        iu = np.triu_indices(len(pts), k=1)
        return dist_matrix[iu]

    @staticmethod
    def calculate_angles(face_landmarks, use_3d: bool = True) -> np.ndarray:
        lm = face_landmarks.landmark
        angles = []
        for a_idx, b_idx, c_idx in FaceMLP.FACE_ANGLE_TRIPLETS:
            a, b, c = lm[a_idx], lm[b_idx], lm[c_idx]
            if use_3d:
                ba = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
                bc = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
            else:
                ba = np.array([a.x-b.x, a.y-b.y])
                bc = np.array([c.x-b.x, c.y-b.y])
            cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles.append(np.arccos(np.clip(cos_a, -1, 1)) / np.pi)
        return np.array(angles)

    @staticmethod
    def calculate_ratios(face_landmarks, use_3d: bool = True) -> np.ndarray:
        lm = face_landmarks.landmark
        ratios = []
        for (a1, a2), (b1, b2) in FaceMLP.RATIO_PAIRS:
            la1, la2, lb1, lb2 = lm[a1], lm[a2], lm[b1], lm[b2]
            if use_3d:
                da = np.sqrt((la1.x-la2.x)**2 + (la1.y-la2.y)**2 + (la1.z-la2.z)**2)
                db = np.sqrt((lb1.x-lb2.x)**2 + (lb1.y-lb2.y)**2 + (lb1.z-lb2.z)**2)
            else:
                da = np.sqrt((la1.x-la2.x)**2 + (la1.y-la2.y)**2)
                db = np.sqrt((lb1.x-lb2.x)**2 + (lb1.y-lb2.y)**2)
            ratios.append(da / (db + 1e-8))
        return np.array(ratios)

    @staticmethod
    def extract_features(face_landmarks, use_3d: bool = True) -> np.ndarray:
        """1265-dim feature vector: distances + angles + ratios"""
        distances = FaceMLP.calculate_distances(face_landmarks, use_3d)
        face_size = FaceMLP.get_face_size(face_landmarks, use_3d)
        distances = distances / face_size
        angles = FaceMLP.calculate_angles(face_landmarks, use_3d)
        ratios = FaceMLP.calculate_ratios(face_landmarks, use_3d)
        return np.concatenate([distances, angles, ratios])

    @staticmethod
    def check_landmark_quality(face_landmarks) -> bool:
        lm = face_landmarks.landmark
        out = sum(1 for idx in FaceMLP.ALL_INDICES
                  if lm[idx].x < -0.5 or lm[idx].x > 1.5 or lm[idx].y < -0.5 or lm[idx].y > 1.5)
        if out > 5:
            return False
        left_eye, right_eye = lm[33], lm[263]
        if np.sqrt((left_eye.x-right_eye.x)**2 + (left_eye.y-right_eye.y)**2) < 0.02:
            return False
        nose, chin = lm[1], lm[152]
        if np.sqrt((nose.x-chin.x)**2 + (nose.y-chin.y)**2) < 0.01:
            return False
        return True

    def train(self, model_data: Dict, epochs: int = 300, batch_size: int = 32,
              verbose: bool = True, progress_callback=None) -> Optional[Dict]:
        expressions = model_data.get('expressions', {})
        if len(expressions) < 2:
            if verbose:
                print(f"ต้องมีอย่างน้อย 2 expressions แต่มี {len(expressions)}")
            return None

        X_list, y_list = [], []
        self.classes = sorted(expressions.keys())
        class_to_idx = {n: i for i, n in enumerate(self.classes)}

        for name, samples in expressions.items():
            for sample in samples:
                if len(sample) == self.NUM_DISTANCES:
                    feature = np.concatenate([np.array(sample), np.zeros(self.NUM_ANGLES + self.NUM_RATIOS)])
                elif len(sample) == self.NUM_DISTANCES + self.NUM_ANGLES:
                    feature = np.concatenate([np.array(sample), np.zeros(self.NUM_RATIOS)])
                elif len(sample) == self.INPUT_SIZE:
                    feature = np.array(sample)
                else:
                    continue
                X_list.append(feature)
                y_list.append(class_to_idx[name])

        return self.train_from_data(
            X_list, y_list, self.classes,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, progress_callback=progress_callback
        )

    def predict_from_landmarks(self, face_landmarks) -> Tuple[str, float]:
        trained_dim = len(self.feature_mean) if self.feature_mean is not None else None
        distances = self.calculate_distances(face_landmarks, self.use_3d)
        face_size = self.get_face_size(face_landmarks, self.use_3d)
        distances = distances / face_size
        angles = self.calculate_angles(face_landmarks, self.use_3d)
        ratios = self.calculate_ratios(face_landmarks, self.use_3d)

        if trained_dim == self.NUM_DISTANCES:
            features = distances
        elif trained_dim == self.NUM_DISTANCES + self.NUM_ANGLES:
            features = np.concatenate([distances, angles])
        else:
            features = np.concatenate([distances, angles, ratios])

        return self.predict(features)

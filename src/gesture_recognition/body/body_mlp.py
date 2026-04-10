"""
Body Pose MLP Classifier
==========================
MLP classifier สำหรับจำแนกท่าทาง (pose) ของร่างกาย
ใช้ MediaPipe Pose 33 landmarks → 548 features (528 distances + 20 angles)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from gesture_recognition.core.base_mlp import BaseMLP


class BodyMLP(BaseMLP):
    """
    MLP Classifier สำหรับจำแนกท่าทางร่างกาย

    Features:
    - 528 pairwise distances (33 landmarks → C(33,2) = 528)
    - 20 joint angles (key body joints)
    - Total: 548 dims

    Architecture: Input(548) → Hidden(256, ReLU) → Hidden(128, ReLU) → Output(softmax)
    """

    # 20 angle triplets สำหรับ key body joints
    # Format: (A, B, C) → angle at B between vectors BA and BC
    ANGLE_TRIPLETS = [
        # Shoulders
        (12, 11, 13),   # Left shoulder angle
        (11, 12, 14),   # Right shoulder angle
        # Elbows
        (11, 13, 15),   # Left elbow flexion
        (12, 14, 16),   # Right elbow flexion
        # Wrists
        (13, 15, 19),   # Left wrist angle
        (14, 16, 20),   # Right wrist angle
        # Hips
        (12, 24, 26),   # Right hip angle
        (11, 23, 25),   # Left hip angle
        # Knees
        (23, 25, 27),   # Left knee flexion
        (24, 26, 28),   # Right knee flexion
        # Ankles
        (25, 27, 31),   # Left ankle angle
        (26, 28, 32),   # Right ankle angle
        # Spine / Torso
        (11, 23, 25),   # Left torso-hip alignment
        (12, 24, 26),   # Right torso-hip alignment
        (13, 11, 23),   # Left arm-torso angle
        (14, 12, 24),   # Right arm-torso angle
        # Cross-body
        (11, 0, 12),    # Neck/head angle (shoulders through nose)
        (23, 0, 24),    # Hips through nose (body lean)
        (15, 11, 23),   # Left arm to left hip (arm raise)
        (16, 12, 24),   # Right arm to right hip (arm raise)
    ]

    NUM_LANDMARKS = 33
    NUM_DISTANCES = NUM_LANDMARKS * (NUM_LANDMARKS - 1) // 2  # 528
    NUM_ANGLES = len(ANGLE_TRIPLETS)  # 20
    INPUT_SIZE = NUM_DISTANCES + NUM_ANGLES  # 548

    # Body size = distance between left shoulder (11) and right shoulder (12)
    # pair index for (11, 12) in upper-triangular ordering:
    # index = sum_{k=0}^{10} (33-1-k) + (12 - 11 - 1) = sum_{k=0}^{10}(32-k) + 0
    # = (32+22)*11/2 = 297
    BODY_SIZE_INDEX = 297

    def __init__(self, input_size: int = None, hidden_sizes: List[int] = None,
                 learning_rate: float = 0.001):
        if input_size is None:
            input_size = self.INPUT_SIZE
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        super().__init__(input_size, hidden_sizes, learning_rate)

    # ---- Feature extraction ----

    @staticmethod
    def _get_pts(landmarks) -> List[Tuple[float, float, float]]:
        """Extract (x, y, z) from landmarks (supports both list and landmark object)"""
        if hasattr(landmarks, 'landmark'):
            return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        return [(lm.x, lm.y, lm.z) for lm in landmarks]

    @staticmethod
    def calculate_distances(landmarks) -> np.ndarray:
        """Pairwise distances ระหว่าง 33 landmarks → 528 features (vectorized)"""
        pts = np.asarray(BodyMLP._get_pts(landmarks))
        diff = pts[:, None, :] - pts[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
        iu = np.triu_indices(len(pts), k=1)
        return dist_matrix[iu]

    @staticmethod
    def calculate_angles(landmarks) -> np.ndarray:
        """Joint angles จาก landmarks → 20 features (normalized to [0,1])"""
        pts = BodyMLP._get_pts(landmarks)
        angles = []
        for a_idx, b_idx, c_idx in BodyMLP.ANGLE_TRIPLETS:
            a = np.array(pts[a_idx])
            b = np.array(pts[b_idx])
            c = np.array(pts[c_idx])

            ba = a - b
            bc = c - b

            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) / np.pi  # Normalize to [0, 1]
            angles.append(angle)

        return np.array(angles)

    @staticmethod
    def normalize_distances_by_body_size(distances: np.ndarray) -> np.ndarray:
        """Normalize distances ด้วย shoulder width (body size)"""
        body_size = distances[BodyMLP.BODY_SIZE_INDEX] + 1e-8
        return distances / body_size

    @staticmethod
    def extract_features(landmarks) -> np.ndarray:
        """Extract full 548-dim feature vector: 528 distances + 20 angles"""
        distances = BodyMLP.calculate_distances(landmarks)
        distances = BodyMLP.normalize_distances_by_body_size(distances)
        angles = BodyMLP.calculate_angles(landmarks)
        return np.concatenate([distances, angles])

    @staticmethod
    def check_landmark_quality(landmarks) -> bool:
        """ตรวจสอบคุณภาพ pose landmarks"""
        pts = BodyMLP._get_pts(landmarks)
        if len(pts) < 33:
            return False

        # Check visibility of key landmarks (shoulders, hips)
        key_indices = [11, 12, 23, 24]  # shoulders + hips
        if hasattr(landmarks, 'landmark'):
            for idx in key_indices:
                lm = landmarks.landmark[idx]
                if hasattr(lm, 'visibility') and lm.visibility < 0.3:
                    return False

        # Check shoulder width is reasonable
        l_shoulder = np.array(pts[11])
        r_shoulder = np.array(pts[12])
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        if shoulder_width < 0.02:
            return False

        return True

    # ---- Training ----

    def train(self, model_data: Dict, epochs: int = 300,
              batch_size: int = 32, verbose: bool = True,
              progress_callback=None) -> Optional[Dict]:
        """Train MLP จาก pose model JSON"""
        poses = model_data.get('poses', {})

        if len(poses) < 2:
            if verbose:
                print(f"ต้องมีอย่างน้อย 2 ท่าทาง แต่มี {len(poses)} ท่าทาง")
            return None

        X_list = []
        y_list = []
        self.classes = sorted(poses.keys())
        class_to_idx = {name: i for i, name in enumerate(self.classes)}

        for name, samples in poses.items():
            for sample in samples:
                if len(sample) == self.INPUT_SIZE:
                    # Full format: 528 distances + 20 angles
                    dists = np.array(sample[:self.NUM_DISTANCES])
                    angles = np.array(sample[self.NUM_DISTANCES:self.INPUT_SIZE])
                    body_size = dists[self.BODY_SIZE_INDEX] + 1e-8
                    dists = dists / body_size
                    feature = np.concatenate([dists, angles])
                elif len(sample) == self.NUM_DISTANCES:
                    # Old format: distances only, pad zeros for angles
                    dists = np.array(sample)
                    body_size = dists[self.BODY_SIZE_INDEX] + 1e-8
                    dists = dists / body_size
                    feature = np.concatenate([dists, np.zeros(self.NUM_ANGLES)])
                else:
                    continue

                X_list.append(feature)
                y_list.append(class_to_idx[name])

        if len(X_list) < 2:
            if verbose:
                print("ข้อมูลไม่เพียงพอ")
            return None

        return self.train_from_data(
            X_list, y_list, self.classes,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, progress_callback=progress_callback
        )

    # ---- Prediction ----

    def predict_from_landmarks(self, landmarks) -> Tuple[str, float]:
        """ทำนายท่าทางร่างกายจาก MediaPipe Pose landmarks"""
        features = self.extract_features(landmarks)
        return self.predict(features)

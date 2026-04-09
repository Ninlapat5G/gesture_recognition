"""
Temporal Smoother — ลด flickering ด้วย weighted voting
ใช้ deque สำหรับ O(1) append/pop, รองรับ source key แบบ dynamic
"""

from collections import deque, defaultdict
from typing import Tuple


class TemporalSmoother:
    """
    Temporal smoothing เพื่อลด flickering ของผล recognition
    ใช้ voting window จาก N frames ล่าสุด

    รองรับ source key อะไรก็ได้: 'left', 'right', 'both', 'face', 'body', ...
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(self, source: str, gesture: str, confidence: float):
        """เพิ่มผลทำนายล่าสุด"""
        self._history[source].append((gesture, confidence))

    def get_smoothed(self, source: str) -> Tuple[str, float]:
        """ดึงผลที่ smooth แล้ว ใช้ weighted voting"""
        history = self._history[source]

        if not history:
            return "", 0.0

        votes = {}
        for i, (gesture, confidence) in enumerate(history):
            weight = i + 1
            if gesture not in votes:
                votes[gesture] = {'weight': 0, 'conf_sum': 0}
            votes[gesture]['weight'] += weight
            votes[gesture]['conf_sum'] += confidence * weight

        best_gesture = max(votes, key=lambda g: votes[g]['weight'])
        total_weight = votes[best_gesture]['weight']
        avg_conf = votes[best_gesture]['conf_sum'] / total_weight if total_weight > 0 else 0

        return best_gesture, avg_conf

    def reset(self, source: str = None):
        """ล้าง history (ถ้าไม่ระบุ source จะล้างทั้งหมด)"""
        if source:
            self._history[source].clear()
        else:
            self._history.clear()

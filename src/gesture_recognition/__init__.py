"""
Gesture Recognition — lightweight hand, body, and face classifiers on MediaPipe.

Quick start::

    from gesture_recognition.hand import HandRecognition
    from gesture_recognition.body import BodyRecognition
    from gesture_recognition.face import FaceRecognition
"""

__version__ = "0.0.2"

from gesture_recognition.hand.hand_recognition import HandRecognition
from gesture_recognition.hand.hand_mlp import HandMLP
from gesture_recognition.body.body_recognition import BodyRecognition
from gesture_recognition.body.body_mlp import BodyMLP
from gesture_recognition.face.face_recognition import FaceRecognition
from gesture_recognition.face.face_mlp import FaceMLP
from gesture_recognition.core.base_mlp import BaseMLP
from gesture_recognition.core.temporal_smoother import TemporalSmoother

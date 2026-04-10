"""
Real-time hand gesture recognition from a webcam.

Usage:
    1. Capture a gesture model via `python run_hand.py` and save it.
    2. Point MODEL_PATH at the resulting JSON file.
    3. Run:  python example/hand_webcam.py
    4. Press ESC to quit.
"""

import sys
import os
import cv2

# Allow running this file directly from the repo root (before pip install)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gesture_recognition.hand.hand_recognition import HandRecognition


MODEL_PATH = "my_gestures.json"
# Optional: point at a Thai-capable TTF if you want Thai labels drawn on screen.
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    rec = HandRecognition(
        font_path=FONT_PATH,
        mode="mae",          # switch to "mlp" after training an MLP
        smoothing_window=5,  # 0 disables temporal smoothing
    )
    model = rec.load_model(MODEL_PATH)

    # If you've trained an MLP, uncomment these two lines to use it:
    # rec.load_mlp(MODEL_PATH)
    # rec.set_mode("mlp")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        status, gestures, confidences, annotated = rec.predict_frame(
            model, frame, show_gesture=1
        )

        cv2.imshow("Hand Gesture Recognition  [ESC to quit]", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

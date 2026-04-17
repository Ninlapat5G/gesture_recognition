"""
Real-time hand gesture recognition from a webcam.

Setup (one-time):
    pip install -e .          # from the repo root
    python run_gui.py hand    # capture gestures and save as my_gestures.json

Run:
    python example/hand_webcam.py

Press ESC to quit.
"""

import sys
import cv2

from gesture_recognition import HandRecognition


MODEL_PATH = "my_gestures.json"
# Optional: point at a Thai-capable TTF if you want Thai labels drawn on screen.
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    # mode="mae"  → works instantly after capturing 2-3 samples per gesture.
    # mode="mlp"  → more accurate, but requires training first in the GUI.
    rec = HandRecognition(
        font_path=FONT_PATH,
        mode="mae",
        smoothing_window=5,  # 0 disables temporal smoothing
    )

    try:
        model = rec.load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # If you've trained an MLP, uncomment these two lines to use it:
    # rec.load_mlp(MODEL_PATH)
    # rec.set_mode("mlp")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam", file=sys.stderr)
        sys.exit(1)

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

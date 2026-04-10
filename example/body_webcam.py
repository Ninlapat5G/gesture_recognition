"""
Real-time body pose recognition from a webcam.

Usage:
    1. Capture a pose model via `python run_body.py` and save it.
    2. Point MODEL_PATH at the resulting JSON file.
    3. Run:  python example/body_webcam.py
    4. Press ESC to quit.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gesture_recognition.body.body_recognition import BodyRecognition


MODEL_PATH = "my_poses.json"
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    rec = BodyRecognition(
        font_path=FONT_PATH,
        mode="mae",          # switch to "mlp" once you've trained one
        smoothing_window=5,
    )
    model = rec.load_model(MODEL_PATH)

    # If you've trained an MLP, uncomment to use it:
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

        pose_name, confidence, annotated = rec.predict_frame(
            model, frame, show_pose=1
        )

        cv2.imshow("Body Pose Recognition  [ESC to quit]", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    rec.close()


if __name__ == "__main__":
    main()

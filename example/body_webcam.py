"""
Real-time body pose recognition from a webcam.

Setup (one-time):
    pip install -e .
    python run_gui.py body    # capture poses and save as my_poses.json

Run:
    python example/body_webcam.py

Press ESC to quit.
"""

import sys
import cv2

from gesture_recognition import BodyRecognition


MODEL_PATH = "my_poses.json"
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    # mode="mae" works instantly; switch to "mlp" once you've trained one.
    rec = BodyRecognition(
        font_path=FONT_PATH,
        mode="mae",
        smoothing_window=5,
    )

    try:
        model = rec.load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # If you've trained an MLP, uncomment to use it:
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

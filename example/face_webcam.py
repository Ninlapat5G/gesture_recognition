"""
Real-time face expression recognition from a webcam.

Usage:
    1. Capture a face model via `python run_face.py` and save it.
    2. Point MODEL_PATH at the resulting JSON file.
    3. Run:  python example/face_webcam.py
    4. Press ESC to quit.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face.face_recognition import FaceRecognition


MODEL_PATH = "my_expressions.json"
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    rec = FaceRecognition(
        font_path=FONT_PATH,
        mode="mae",          # switch to "mlp" once you've trained one
        smoothing_window=5,
        use_3d=True,
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

        expression, confidence, annotated = rec.predict_frame(
            model, frame, show_expression=1
        )

        cv2.imshow("Face Expression Recognition  [ESC to quit]", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

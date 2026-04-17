"""
Real-time face expression recognition from a webcam.

Setup (one-time):
    pip install -e .
    python run_gui.py face    # capture expressions and save as my_expressions.json

Run:
    python example/face_webcam.py

Press ESC to quit.
"""

import sys
import cv2

from gesture_recognition import FaceRecognition


MODEL_PATH = "my_expressions.json"
FONT_PATH = None  # e.g. "C:/Windows/Fonts/Tahoma.ttf"


def main():
    # mode="mae" works instantly; switch to "mlp" once you've trained one.
    rec = FaceRecognition(
        font_path=FONT_PATH,
        mode="mae",
        smoothing_window=5,
        use_3d=True,
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

"""
Run hand recognition on a single image file.

Setup:
    pip install -e .
    python run_gui.py hand    # capture gestures and save as my_gestures.json

Run:
    python example/hand_image.py path/to/image.jpg
"""

import sys
import cv2

from gesture_recognition import HandRecognition


MODEL_PATH = "my_gestures.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: python example/hand_image.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    rec = HandRecognition(mode="mae", smoothing_window=0)
    try:
        model = rec.load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    status, gestures, confidences, annotated = rec.predict_frame(
        model, image, show_gesture=1
    )

    print(f"Status     : {status}")
    print(f"Gestures   : {gestures}")
    print(f"Confidences: {[f'{c:.1f}%' for c in confidences]}")

    cv2.imshow("Hand Recognition  [press any key to close]", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

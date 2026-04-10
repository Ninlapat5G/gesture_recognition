"""
Run hand gesture recognition on a single image file.

Usage:
    python example/hand_image.py path/to/image.jpg
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gesture_recognition.hand.hand_recognition import HandRecognition


MODEL_PATH = "my_gestures.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: python example/hand_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    rec = HandRecognition(mode="mae", smoothing_window=0)
    model = rec.load_model(MODEL_PATH)

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

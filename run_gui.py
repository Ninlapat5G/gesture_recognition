"""Launch one of the bundled capture/train GUIs.

Usage:
    python run_gui.py hand      # Hand gesture recognition
    python run_gui.py body      # Body pose recognition
    python run_gui.py face      # Face expression recognition

After  pip install -e ".[gui]"  you can also run the console scripts:

    gesture-hand
    gesture-body
    gesture-face

This file is a thin convenience wrapper so the repo works with just a
git clone + pip install of the runtime deps (no editable install needed).
"""

import argparse
import os
import sys


def _ensure_src_on_path():
    """Allow running without `pip install -e .` by adding src/ to sys.path."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the hand, body, or face recognition GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "module",
        choices=("hand", "body", "face"),
        help="Which recognizer to launch.",
    )
    args = parser.parse_args()

    _ensure_src_on_path()

    try:
        import customtkinter as ctk
    except ImportError:
        print(
            "customtkinter is not installed.\n"
            "Install the GUI extras:  pip install -e \".[gui]\"\n"
            "or:  pip install customtkinter==5.2.2",
            file=sys.stderr,
        )
        return 1

    if args.module == "hand":
        from gesture_recognition.hand.hand_gui import GestureModelManager as App
    elif args.module == "body":
        from gesture_recognition.body.body_gui import BodyPoseManager as App
    else:
        from gesture_recognition.face.face_gui import FaceExpressionManager as App

    root = ctk.CTk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

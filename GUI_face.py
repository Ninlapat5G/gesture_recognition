"""Entry point for Face Expression Recognition GUI (dev shortcut)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import customtkinter as ctk
from gesture_recognition.face.face_gui import FaceExpressionManager

if __name__ == "__main__":
    root = ctk.CTk()
    FaceExpressionManager(root)
    root.mainloop()

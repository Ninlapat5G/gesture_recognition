"""Entry point for Hand Gesture Recognition GUI (dev shortcut)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import customtkinter as ctk
from gesture_recognition.hand.hand_gui import GestureModelManager

if __name__ == "__main__":
    root = ctk.CTk()
    GestureModelManager(root)
    root.mainloop()

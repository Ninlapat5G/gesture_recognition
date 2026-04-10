"""Entry point for Body Pose Recognition GUI (dev shortcut)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import customtkinter as ctk
from gesture_recognition.body.body_gui import BodyPoseManager

if __name__ == "__main__":
    root = ctk.CTk()
    BodyPoseManager(root)
    root.mainloop()

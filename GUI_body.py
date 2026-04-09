"""Entry point สำหรับ Body Pose Recognition GUI"""
import customtkinter as ctk
from body.body_gui import BodyPoseManager

if __name__ == "__main__":
    root = ctk.CTk()
    app = BodyPoseManager(root)
    root.mainloop()

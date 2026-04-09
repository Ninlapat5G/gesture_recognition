"""Entry point สำหรับ Hand Gesture Recognition GUI"""
import customtkinter as ctk
from hand.hand_gui import GestureModelManager

if __name__ == "__main__":
    root = ctk.CTk()
    app = GestureModelManager(root)
    root.mainloop()

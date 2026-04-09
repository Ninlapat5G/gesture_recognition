"""Entry point สำหรับ Face Expression Recognition GUI"""
import customtkinter as ctk
from face.face_gui import FaceExpressionManager

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceExpressionManager(root)
    root.mainloop()

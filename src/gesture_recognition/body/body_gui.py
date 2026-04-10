"""
Body Pose Recognition GUI
============================
GUI สำหรับจัดการโมเดลท่าทางร่างกาย
ใช้ BodyRecognition API สำหรับ recognition
ไม่มี capture/countdown — ใช้ dataset folder import เป็นหลัก
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import os

from gesture_recognition.body.body_mlp import BodyMLP
from gesture_recognition.body.body_recognition import BodyRecognition
from gesture_recognition.core.temporal_smoother import TemporalSmoother
from gesture_recognition.core.utils import (load_font, get_asset_path, draw_thai_text,
                         display_frame_on_canvas,
                         show_error_popup, show_success_popup)


class BodyPoseManager:
    def __init__(self, window):
        self.window = window
        self.window.title("ระบบจัดการท่าทางร่างกาย")
        self.window.geometry("1400x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # MediaPipe (for video display)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Font
        font_path = get_asset_path("AmericanoDemo.ttf")
        self.thai_font = load_font(font_path, 36)
        self.thai_font_video = load_font(font_path, 28)

        # Recognition API
        self.recognition = BodyRecognition(mode="mae")

        # State
        self.current_model_file = None
        self.poses = {}
        self.selected_pose_name = None

        # Video
        self.frame = None
        self.pose_landmarks = None
        self.pose_detected = False
        self.running = True
        self.show_skeleton = True
        self.show_text_overlay = True
        self.canvas_width = 640
        self.canvas_height = 480
        self.lock = threading.Lock()

        # Recognition
        self.recognition_mode = "mae"
        self.use_smoothing = True
        self.smoother = TemporalSmoother(window_size=5)

        self.create_main_layout()

        self.window.update_idletasks()
        self.canvas_width = max(self.canvas.winfo_width(), 640)
        self.canvas_height = max(self.canvas.winfo_height(), 480)

        self.capture_thread = threading.Thread(target=self.capture_video, daemon=True)
        self.capture_thread.start()
        self.process_thread = threading.Thread(target=self.process_frame, daemon=True)
        self.process_thread.start()

        self.update_gui()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    # =========================================================================
    # Layout
    # =========================================================================

    def create_main_layout(self):
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header.pack(fill="x")

        ctk.CTkLabel(header, text="ระบบจัดการท่าทางร่างกาย",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(side="left", padx=30, pady=20)

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=30, pady=20)

        ctk.CTkButton(btn_frame, text="สร้างโมเดล", command=self.create_new_model,
                      width=130, height=40, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="เปิดโมเดล", command=self.load_model,
                      width=130, height=40, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)

        # Content
        content = ctk.CTkFrame(self.main_container)
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        content.grid_columnconfigure(0, weight=7, minsize=400)
        content.grid_columnconfigure(1, weight=3, minsize=300)
        content.grid_rowconfigure(0, weight=1)

        # Video panel
        video_cont = ctk.CTkFrame(content, corner_radius=15)
        video_cont.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        video_cont.grid_rowconfigure(1, weight=1)
        video_cont.grid_columnconfigure(0, weight=1)

        info = ctk.CTkFrame(video_cont, fg_color="transparent")
        info.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        ctk.CTkLabel(info, text="กล้องสด — Body Pose",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=5)
        self.model_name_label = ctk.CTkLabel(info, text="", font=ctk.CTkFont(size=14), text_color="gray")
        self.model_name_label.pack(pady=2)

        self.canvas = ctk.CTkCanvas(video_cont, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

        # Right panel (scrollable)
        ctrl = ctk.CTkScrollableFrame(content, corner_radius=15, fg_color=("gray90", "gray13"))
        ctrl.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")

        # Pose display
        pf = ctk.CTkFrame(ctrl, corner_radius=10)
        pf.pack(padx=10, pady=(10, 15), fill="x")
        pc = ctk.CTkFrame(pf, corner_radius=8)
        pc.pack(padx=15, pady=15, fill="x")
        ctk.CTkLabel(pc, text="ท่าทางร่างกาย", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#e67e22").pack(pady=(10, 5))
        self.pose_label = ctk.CTkLabel(pc, text="ไม่พบร่างกาย", font=ctk.CTkFont(size=20))
        self.pose_label.pack(pady=(0, 10))

        # Settings
        sf = ctk.CTkFrame(ctrl, corner_radius=10)
        sf.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(sf, text="การตั้งค่า", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        self.skeleton_switch = ctk.CTkSwitch(sf, text="แสดง Skeleton", font=ctk.CTkFont(size=14),
                                              command=lambda: setattr(self, 'show_skeleton', self.skeleton_switch.get()))
        self.skeleton_switch.pack(padx=15, pady=8)
        self.skeleton_switch.select()

        self.text_switch = ctk.CTkSwitch(sf, text="แสดงข้อความบนวิดีโอ", font=ctk.CTkFont(size=14),
                                          command=lambda: setattr(self, 'show_text_overlay', self.text_switch.get()))
        self.text_switch.pack(padx=15, pady=8)
        self.text_switch.select()

        self.smoothing_switch = ctk.CTkSwitch(sf, text="Temporal Smoothing", font=ctk.CTkFont(size=14),
                                               command=self.toggle_smoothing)
        self.smoothing_switch.pack(padx=15, pady=(8, 15))
        self.smoothing_switch.select()

        # MLP
        mf = ctk.CTkFrame(ctrl, corner_radius=10)
        mf.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(mf, text="MLP Neural Network", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        mc = ctk.CTkFrame(mf, fg_color="transparent")
        mc.pack(padx=15, pady=(0, 8), fill="x")
        ctk.CTkLabel(mc, text="โหมด:", font=ctk.CTkFont(size=14)).pack(side="left", padx=(0, 10))
        self.mode_var = ctk.StringVar(value="MAE (ดั้งเดิม)")
        ctk.CTkSegmentedButton(mc, values=["MAE (ดั้งเดิม)", "MLP (Neural)"],
                                variable=self.mode_var, command=self.switch_recognition_mode,
                                font=ctk.CTkFont(size=12)).pack(side="left", fill="x", expand=True)

        ctk.CTkButton(mf, text="Train MLP", font=ctk.CTkFont(size=14, weight="bold"),
                      height=40, command=self.train_mlp_model,
                      fg_color="#8e44ad", hover_color="#7d3c98").pack(padx=15, pady=(0, 8), fill="x")
        self.mlp_status_label = ctk.CTkLabel(mf, text="ยังไม่ได้ train MLP",
                                              font=ctk.CTkFont(size=12), text_color="gray")
        self.mlp_status_label.pack(padx=15, pady=(0, 15))

        # Actions — Dataset import only (no capture button)
        af = ctk.CTkFrame(ctrl, corner_radius=10)
        af.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(af, text="การดำเนินการ", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        ctk.CTkButton(af, text="นำเข้าจาก Dataset",
                      font=ctk.CTkFont(size=16, weight="bold"), height=50,
                      command=self.load_dataset_folder,
                      fg_color="#27ae60", hover_color="#219a52").pack(padx=15, pady=(0, 10), fill="x")

        # Info text
        info_text = ("เลือก folder ที่มี subfolder\n"
                     "ตั้งชื่อตามท่าทาง เช่น:\n"
                     "  dataset/\n"
                     "    ├── standing/\n"
                     "    │   ├── img1.jpg\n"
                     "    │   └── img2.png\n"
                     "    └── sitting/\n"
                     "        └── img1.jpg")
        ctk.CTkLabel(af, text=info_text, font=ctk.CTkFont(size=11, family="Consolas"),
                     justify="left", text_color="gray").pack(padx=15, pady=(0, 10))

        ctk.CTkButton(af, text="ลบท่าทางทั้งหมด", font=ctk.CTkFont(size=15), height=40,
                      command=self.clear_poses,
                      fg_color="#e74c3c", hover_color="#c0392b").pack(padx=15, pady=(0, 15), fill="x")

        # Stats
        stf = ctk.CTkFrame(ctrl, corner_radius=10)
        stf.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(stf, text="สถิติ", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))
        self.stats_label = ctk.CTkLabel(stf, text=self.get_stats_text(),
                                         font=ctk.CTkFont(size=14), justify="left")
        self.stats_label.pack(padx=15, pady=(0, 15))

        # Pose list
        lf = ctk.CTkFrame(ctrl, corner_radius=10)
        lf.pack(padx=10, pady=(0, 15), fill="both", expand=True)
        hf = ctk.CTkFrame(lf, fg_color="transparent")
        hf.pack(fill="x", pady=(15, 10), padx=15)
        ctk.CTkLabel(hf, text="ท่าทางที่บันทึกไว้", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        self.pose_list_container = ctk.CTkScrollableFrame(lf, height=250, fg_color=("gray85", "gray20"))
        self.pose_list_container.pack(padx=15, pady=(0, 15), fill="both", expand=True)

        self.update_pose_list()

    # =========================================================================
    # Settings
    # =========================================================================

    def toggle_smoothing(self):
        self.use_smoothing = self.smoothing_switch.get()
        if not self.use_smoothing:
            self.smoother.reset()

    def switch_recognition_mode(self, value):
        if "MLP" in value:
            if not self.recognition.body_mlp.is_trained:
                show_error_popup(self.window, "ยังไม่ได้ Train MLP", "กรุณา Train MLP ก่อนใช้โหมดนี้")
                self.mode_var.set("MAE (ดั้งเดิม)")
                return
            self.recognition_mode = "mlp"
        else:
            self.recognition_mode = "mae"
        self.recognition.set_mode(self.recognition_mode)
        self.smoother.reset()

    # =========================================================================
    # Model I/O
    # =========================================================================

    def create_new_model(self):
        dialog = ctk.CTkInputDialog(text="ชื่อไฟล์โมเดล (ไม่ต้องใส่ .json):", title="สร้างโมเดลใหม่")
        filename = dialog.get_input()
        if not filename:
            return
        if not filename.endswith('.json'):
            filename += '.json'
        if os.path.exists(filename):
            show_error_popup(self.window, "ข้อผิดพลาด", "ไฟล์นี้มีอยู่แล้ว")
            return
        self.current_model_file = filename
        self.poses = {}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"poses": {}}, f, ensure_ascii=False, indent=2)
        self.model_name_label.configure(text=filename)
        self.stats_label.configure(text=self.get_stats_text())
        self.update_pose_list()
        show_success_popup(self.window, "สร้างโมเดลสำเร็จ")

    def load_model(self):
        filename = filedialog.askopenfilename(title="เลือกไฟล์โมเดล",
                                               filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.poses = data.get('poses', data) if isinstance(data, dict) else {}
            if not isinstance(self.poses, dict):
                self.poses = {}
            self.current_model_file = filename
            self.model_name_label.configure(text=os.path.basename(filename))
            self.stats_label.configure(text=self.get_stats_text())
            self.update_pose_list()

            # Auto-load MLP
            base = filename.replace('.json', '')
            mlp_path = f"{base}_body.mlp.json"
            if os.path.exists(mlp_path) and self.recognition.body_mlp.load(mlp_path):
                self.mlp_status_label.configure(
                    text=f"MLP พร้อม: {len(self.recognition.body_mlp.classes)} ท่า", text_color="#2ecc71")
            else:
                self.mlp_status_label.configure(text="ยังไม่ได้ train MLP", text_color="gray")

            show_success_popup(self.window, "โหลดโมเดลสำเร็จ")
        except Exception as e:
            show_error_popup(self.window, "ข้อผิดพลาด", str(e))

    def save_model_to_file(self):
        if self.current_model_file:
            with open(self.current_model_file, 'w', encoding='utf-8') as f:
                json.dump({"poses": self.poses}, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Dataset import (primary action)
    # =========================================================================

    def load_dataset_folder(self):
        if not self.current_model_file:
            show_error_popup(self.window, "ไม่มีโมเดล", "กรุณาสร้างหรือเปิดโมเดลก่อน")
            return

        folder = filedialog.askdirectory(title="เลือก folder dataset (subfolder = ชื่อท่าทาง)")
        if not folder:
            return

        popup = ctk.CTkToplevel(self.window)
        popup.title("นำเข้า Dataset")
        popup.geometry("500x250")
        popup.transient(self.window)
        popup.grab_set()

        ctk.CTkLabel(popup, text="กำลังประมวลผล Body Pose...",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10))
        progress = ctk.CTkProgressBar(popup, width=400)
        progress.pack(pady=10, padx=40)
        progress.set(0)
        status = ctk.CTkLabel(popup, text="เริ่มต้น...", font=ctk.CTkFont(size=14))
        status.pack(pady=5)
        detail = ctk.CTkLabel(popup, text="", font=ctk.CTkFont(size=12), text_color="gray")
        detail.pack(pady=5)

        def do_import():
            detected_count = [0]
            skipped_count = [0]

            def cb(current, total, class_name):
                try:
                    progress.set(current / total if total > 0 else 0)
                    status.configure(text=f"{class_name}: {current}/{total} รูป")
                except Exception:
                    pass

            result = self.recognition.load_dataset(folder, progress_cb=cb)

            for name, features_list in result.items():
                if name not in self.poses:
                    self.poses[name] = []
                self.poses[name].extend(features_list)
                detected_count[0] += len(features_list)

            self.save_model_to_file()

            try:
                total_added = sum(len(v) for v in result.values())
                num_classes = len(result)
                status.configure(text=f"เสร็จ: {num_classes} ท่า, {total_added} ตัวอย่าง")
                detail.configure(text=f"ท่าทาง: {', '.join(result.keys()) if result else '-'}")
                self.stats_label.configure(text=self.get_stats_text())
                self.update_pose_list()
                popup.after(3000, popup.destroy)
            except Exception:
                pass

        threading.Thread(target=do_import, daemon=True).start()

    # =========================================================================
    # MLP Training
    # =========================================================================

    def train_mlp_model(self):
        if not self.current_model_file:
            show_error_popup(self.window, "ไม่มีโมเดล", "กรุณาสร้างหรือเปิดโมเดลก่อน")
            return
        if len(self.poses) < 2:
            show_error_popup(self.window, "ข้อมูลไม่พอ", "ต้องมีอย่างน้อย 2 ท่าทาง")
            return

        popup = ctk.CTkToplevel(self.window)
        popup.title("Training Body MLP")
        popup.geometry("450x300")
        popup.transient(self.window)
        popup.grab_set()

        ctk.CTkLabel(popup, text="กำลัง Training...",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
        pbar = ctk.CTkProgressBar(popup, width=350)
        pbar.pack(pady=10, padx=40)
        pbar.set(0)
        slabel = ctk.CTkLabel(popup, text="เริ่มต้น...", font=ctk.CTkFont(size=14))
        slabel.pack(pady=5)
        rlabel = ctk.CTkLabel(popup, text="", font=ctk.CTkFont(size=13), justify="left")
        rlabel.pack(pady=10, padx=20)
        close_btn = ctk.CTkButton(popup, text="ปิด", command=popup.destroy,
                                   state="disabled", width=120)
        close_btn.pack(pady=15)

        total_epochs = 300

        def do_train():
            model_data = {"poses": self.poses}

            def cb(epoch, tl, ta, vl, va):
                try:
                    pbar.set((epoch + 1) / total_epochs)
                    slabel.configure(text=f"Epoch {epoch+1}/{total_epochs} | "
                                         f"Train: {ta:.1f}% | Val: {va:.1f}%")
                except Exception:
                    pass

            stats = self.recognition.train_mlp(model_data, epochs=total_epochs,
                                                verbose=False, progress_callback=cb)

            if self.current_model_file:
                self.recognition.save_mlp(self.current_model_file)

            try:
                if stats:
                    slabel.configure(text="Training เสร็จสิ้น!")
                    va = stats.get('final_val_accuracy', 0)
                    ta = stats.get('final_accuracy', 0)
                    rlabel.configure(text=f"Train: {ta:.1f}% | Val: {va:.1f}%")
                    self.mlp_status_label.configure(
                        text=f"MLP พร้อม: {len(self.recognition.body_mlp.classes)} ท่า",
                        text_color="#2ecc71")
                close_btn.configure(state="normal")
                pbar.set(1.0)
            except Exception:
                pass

        threading.Thread(target=do_train, daemon=True).start()

    # =========================================================================
    # Pose list
    # =========================================================================

    def get_stats_text(self):
        count = len(self.poses)
        total = sum(len(v) for v in self.poses.values())
        return f"ท่าทาง: {count} | ตัวอย่าง: {total}"

    def update_pose_list(self):
        for w in self.pose_list_container.winfo_children():
            w.destroy()

        if not self.poses:
            ctk.CTkLabel(self.pose_list_container, text="ยังไม่มีข้อมูล\nนำเข้าจาก Dataset เพื่อเริ่มต้น",
                         font=ctk.CTkFont(size=14), text_color="gray",
                         justify="center").pack(pady=30, expand=True)
            return

        for name in sorted(self.poses.keys()):
            samples = self.poses[name]
            item = ctk.CTkFrame(self.pose_list_container, corner_radius=8, height=40)
            item.pack(padx=5, pady=3, fill="x")
            item.pack_propagate(False)

            ctk.CTkLabel(item, text=f"{name} ({len(samples)} ตัวอย่าง)",
                         font=ctk.CTkFont(size=14)).pack(side="left", padx=10)

            ctk.CTkButton(item, text="ลบ", width=50, height=28,
                          fg_color="#e74c3c", hover_color="#c0392b",
                          command=lambda n=name: self.delete_pose(n)).pack(side="right", padx=5)

    def delete_pose(self, name):
        if name in self.poses:
            del self.poses[name]
            self.save_model_to_file()
            self.stats_label.configure(text=self.get_stats_text())
            self.update_pose_list()

    def clear_poses(self):
        self.poses = {}
        self.save_model_to_file()
        self.stats_label.configure(text=self.get_stats_text())
        self.update_pose_list()

    # =========================================================================
    # Video processing
    # =========================================================================

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            with self.lock:
                self.frame = frame
                if results.pose_landmarks:
                    self.pose_landmarks = results.pose_landmarks
                    self.pose_detected = True
                else:
                    self.pose_landmarks = None
                    self.pose_detected = False

        cap.release()

    def process_frame(self):
        import time
        while self.running:
            with self.lock:
                frame = self.frame
                landmarks = self.pose_landmarks
                detected = self.pose_detected

            if frame is None:
                time.sleep(0.03)
                continue

            display = frame.copy()

            # Draw skeleton
            if self.show_skeleton and detected and landmarks:
                self.mp_drawing.draw_landmarks(
                    display, landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Recognition
            if detected and landmarks and self.poses and self.show_text_overlay:
                model_data = {"poses": self.poses}
                pose_name, confidence = self.recognition.recognize_from_landmarks(landmarks, model_data)

                if self.use_smoothing:
                    self.smoother.update('body', pose_name, confidence)
                    pose_name, confidence = self.smoother.get_smoothed('body')

                if pose_name and confidence > 0:
                    color = (0, 255, 0) if confidence >= 70 else (0, 165, 255)
                    text = f"{pose_name} ({confidence:.1f}%)"
                else:
                    color = (128, 128, 128)
                    text = "ไม่รู้จัก"

                display = draw_thai_text(display, text, (20, 20), self.thai_font_video, color)

                try:
                    self.pose_label.configure(text=text if pose_name else "ไม่รู้จัก")
                except Exception:
                    pass

            display_frame_on_canvas(display, self.canvas, self.canvas_width, self.canvas_height)
            time.sleep(0.03)

    def update_gui(self):
        try:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 50 and ch > 50:
                self.canvas_width = cw
                self.canvas_height = ch
        except Exception:
            pass
        if self.running:
            self.window.after(1000, self.update_gui)

    def on_closing(self):
        self.running = False
        self.pose.close()
        self.recognition.close()
        self.window.destroy()


def _cli_main():
    """Entry point for ``gesture-body`` console script."""
    root = ctk.CTk()
    BodyPoseManager(root)
    root.mainloop()


if __name__ == "__main__":
    _cli_main()

"""
Face Expression Recognition GUI
=================================
GUI สำหรับจัดการโมเดลท่าทางใบหน้า
ใช้ FaceRecognition API สำหรับ recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tkinter import filedialog
import os

from face.face_mlp import FaceMLP
from face.face_recognition import FaceRecognition
from core.temporal_smoother import TemporalSmoother
from core.utils import (load_font, get_asset_path, draw_thai_text,
                         display_frame_on_canvas, draw_countdown_overlay,
                         show_error_popup, show_success_popup)


class FaceExpressionManager:
    def __init__(self, window):
        self.window = window
        self.window.title("ระบบจัดการท่าทางใบหน้า")
        self.window.geometry("1400x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # MediaPipe (for video display)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Font
        font_path = get_asset_path("AmericanoDemo.ttf")
        self.thai_font = load_font(font_path, 36)
        self.thai_font_video = load_font(font_path, 28)

        # Recognition API
        self.recognition = FaceRecognition(mode="mae", use_3d=True)

        # State
        self.current_model_file = None
        self.expressions = {}
        self.selected_expression_name = None

        # Video
        self.frame = None
        self.face_landmarks = None
        self.face_detected = False
        self.running = True
        self.show_mesh = True
        self.show_text_overlay = True
        self.canvas_width = 640
        self.canvas_height = 480
        self.lock = threading.Lock()

        # Recognition
        self.recognition_mode = "mae"
        self.use_smoothing = True
        self.use_3d = True
        self.smoother = TemporalSmoother(window_size=5)

        # Countdown
        self.countdown_active = False
        self.countdown_value = 0
        self.countdown_expression_name = None
        self._countdown_is_new = True

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

        header = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header.pack(fill="x")

        ctk.CTkLabel(header, text="ระบบจัดการท่าทางใบหน้า",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(side="left", padx=30, pady=20)

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=30, pady=20)

        ctk.CTkButton(btn_frame, text="สร้างโมเดล", command=self.create_new_model,
                      width=130, height=40, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="เปิดโมเดล", command=self.load_model,
                      width=130, height=40, font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=5)

        content = ctk.CTkFrame(self.main_container)
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        content.grid_columnconfigure(0, weight=7, minsize=400)
        content.grid_columnconfigure(1, weight=3, minsize=300)
        content.grid_rowconfigure(0, weight=1)

        # Video
        video_cont = ctk.CTkFrame(content, corner_radius=15)
        video_cont.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        video_cont.grid_rowconfigure(1, weight=1)
        video_cont.grid_columnconfigure(0, weight=1)

        info = ctk.CTkFrame(video_cont, fg_color="transparent")
        info.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        ctk.CTkLabel(info, text="กล้องสด — Face Expression",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=5)
        self.model_name_label = ctk.CTkLabel(info, text="", font=ctk.CTkFont(size=14), text_color="gray")
        self.model_name_label.pack(pady=2)

        self.canvas = ctk.CTkCanvas(video_cont, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

        # Right panel
        ctrl = ctk.CTkScrollableFrame(content, corner_radius=15, fg_color=("gray90", "gray13"))
        ctrl.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")

        # Expression display
        ef = ctk.CTkFrame(ctrl, corner_radius=10)
        ef.pack(padx=10, pady=(10, 15), fill="x")
        ec = ctk.CTkFrame(ef, corner_radius=8)
        ec.pack(padx=15, pady=15, fill="x")
        ctk.CTkLabel(ec, text="ท่าทางใบหน้า", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#e67e22").pack(pady=(10, 5))
        self.expression_label = ctk.CTkLabel(ec, text="ไม่พบใบหน้า", font=ctk.CTkFont(size=20))
        self.expression_label.pack(pady=(0, 10))

        # Settings
        sf = ctk.CTkFrame(ctrl, corner_radius=10)
        sf.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(sf, text="การตั้งค่า", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        self.mesh_switch = ctk.CTkSwitch(sf, text="แสดง Face Mesh", font=ctk.CTkFont(size=14),
                                          command=lambda: setattr(self, 'show_mesh', self.mesh_switch.get()))
        self.mesh_switch.pack(padx=15, pady=8); self.mesh_switch.select()

        self.text_switch = ctk.CTkSwitch(sf, text="แสดงข้อความบนวิดีโอ", font=ctk.CTkFont(size=14),
                                          command=lambda: setattr(self, 'show_text_overlay', self.text_switch.get()))
        self.text_switch.pack(padx=15, pady=8); self.text_switch.select()

        self.smoothing_switch = ctk.CTkSwitch(sf, text="Temporal Smoothing", font=ctk.CTkFont(size=14),
                                               command=self.toggle_smoothing)
        self.smoothing_switch.pack(padx=15, pady=8); self.smoothing_switch.select()

        self.use_3d_switch = ctk.CTkSwitch(sf, text="ใช้ 3D (ปิด = 2D)", font=ctk.CTkFont(size=14),
                                            command=self.toggle_3d)
        self.use_3d_switch.pack(padx=15, pady=(8, 15)); self.use_3d_switch.select()

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

        # Actions
        af = ctk.CTkFrame(ctrl, corner_radius=10)
        af.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(af, text="การดำเนินการ", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))

        ctk.CTkButton(af, text="บันทึกท่าทางใหม่ (นับถอยหลัง)",
                      font=ctk.CTkFont(size=16, weight="bold"), height=45,
                      command=self.save_new_expression,
                      fg_color="#3498db", hover_color="#2980b9").pack(padx=15, pady=(0, 10), fill="x")

        ctk.CTkButton(af, text="นำเข้าจาก Dataset",
                      font=ctk.CTkFont(size=16, weight="bold"), height=45,
                      command=self.load_dataset_folder,
                      fg_color="#27ae60", hover_color="#219a52").pack(padx=15, pady=(0, 10), fill="x")

        ctk.CTkButton(af, text="ลบท่าทางทั้งหมด", font=ctk.CTkFont(size=15), height=40,
                      command=self.clear_expressions,
                      fg_color="#e74c3c", hover_color="#c0392b").pack(padx=15, pady=(0, 15), fill="x")

        # Stats
        stf = ctk.CTkFrame(ctrl, corner_radius=10)
        stf.pack(padx=10, pady=(0, 15), fill="x")
        ctk.CTkLabel(stf, text="สถิติ", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(15, 10))
        self.stats_label = ctk.CTkLabel(stf, text=self.get_stats_text(),
                                         font=ctk.CTkFont(size=14), justify="left")
        self.stats_label.pack(padx=15, pady=(0, 15))

        # Selected indicator
        self.selected_frame = ctk.CTkFrame(ctrl, corner_radius=10, fg_color="#2c3e50")
        ctk.CTkLabel(self.selected_frame, text="ท่าทางที่เลือก",
                     font=ctk.CTkFont(size=16, weight="bold"), text_color="#f39c12").pack(pady=(12, 5))
        self.selected_display = ctk.CTkLabel(self.selected_frame, text="",
                                              font=ctk.CTkFont(size=18, weight="bold"), text_color="#ecf0f1")
        self.selected_display.pack(pady=(0, 5))
        ctk.CTkButton(self.selected_frame, text="ยกเลิกการเลือก", command=self.deselect_expression,
                      font=ctk.CTkFont(size=13), height=35, fg_color="#e67e22").pack(padx=15, pady=(5, 12), fill="x")

        # Expression list
        lf = ctk.CTkFrame(ctrl, corner_radius=10)
        lf.pack(padx=10, pady=(0, 15), fill="both", expand=True)
        hf = ctk.CTkFrame(lf, fg_color="transparent")
        hf.pack(fill="x", pady=(15, 10), padx=15)
        ctk.CTkLabel(hf, text="ท่าทางที่บันทึกไว้", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        self.expression_list_container = ctk.CTkScrollableFrame(lf, height=200, fg_color=("gray85", "gray20"))
        self.expression_list_container.pack(padx=15, pady=(0, 10), fill="both", expand=True)

        self.add_sample_button = ctk.CTkButton(lf, text="เพิ่มตัวอย่าง",
                                                font=ctk.CTkFont(size=16, weight="bold"), height=45,
                                                command=self.add_expression_sample,
                                                fg_color="#2ecc71", hover_color="#27ae60", state="disabled")
        self.add_sample_button.pack(padx=15, pady=(0, 15), fill="x")

        self.update_expression_list()

    # =========================================================================
    # Settings
    # =========================================================================

    def toggle_smoothing(self):
        self.use_smoothing = self.smoothing_switch.get()
        if not self.use_smoothing:
            self.smoother.reset()

    def toggle_3d(self):
        self.use_3d = self.use_3d_switch.get()
        self.recognition.use_3d = self.use_3d
        self.recognition.face_mlp.use_3d = self.use_3d

    def switch_recognition_mode(self, value):
        if "MLP" in value:
            if not self.recognition.face_mlp.is_trained:
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
        self.expressions = {}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"expressions": {}}, f, ensure_ascii=False, indent=2)
        self.model_name_label.configure(text=filename)
        self.stats_label.configure(text=self.get_stats_text())
        self.update_expression_list()
        show_success_popup(self.window, "สร้างโมเดลสำเร็จ")

    def load_model(self):
        filename = filedialog.askopenfilename(title="เลือกไฟล์โมเดล",
                                               filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.expressions = data.get('expressions', data) if isinstance(data, dict) else {}
            if not isinstance(self.expressions, dict):
                self.expressions = {}
            self.current_model_file = filename
            self.model_name_label.configure(text=os.path.basename(filename))
            self.stats_label.configure(text=self.get_stats_text())
            self.update_expression_list()

            # Auto-load MLP
            base = filename.replace('.json', '')
            mlp_path = f"{base}_face.mlp.json"
            if os.path.exists(mlp_path) and self.recognition.face_mlp.load(mlp_path):
                self.mlp_status_label.configure(
                    text=f"MLP พร้อม: {len(self.recognition.face_mlp.classes)} ท่า", text_color="#2ecc71")
            else:
                self.mlp_status_label.configure(text="ยังไม่ได้ train MLP", text_color="gray")

            show_success_popup(self.window, "โหลดโมเดลสำเร็จ")
        except Exception as e:
            show_error_popup(self.window, "ข้อผิดพลาด", str(e))

    def save_model_to_file(self):
        if self.current_model_file:
            with open(self.current_model_file, 'w', encoding='utf-8') as f:
                json.dump({"expressions": self.expressions}, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Expression capture
    # =========================================================================

    def save_new_expression(self):
        if not self.current_model_file:
            show_error_popup(self.window, "ไม่มีโมเดล", "กรุณาสร้างหรือเปิดโมเดลก่อน")
            return
        dialog = ctk.CTkInputDialog(text="ชื่อท่าทางใบหน้า:", title="บันทึกท่าทางใหม่")
        name = dialog.get_input()
        if not name:
            return
        if name in self.expressions:
            show_error_popup(self.window, "ซ้ำ", f"ท่าทาง '{name}' มีอยู่แล้ว")
            return
        self.countdown_expression_name = name
        self._countdown_is_new = True
        self.start_countdown()

    def add_expression_sample(self):
        if not self.selected_expression_name:
            return
        self.countdown_expression_name = self.selected_expression_name
        self._countdown_is_new = False
        self.start_countdown()

    def start_countdown(self):
        self.countdown_active = True
        self.countdown_value = 3
        self._countdown_tick()

    def _countdown_tick(self):
        if self.countdown_value > 0:
            self.countdown_value -= 1
            self.window.after(1000, self._countdown_tick)
        else:
            self.countdown_active = False
            self._capture_face()

    def _capture_face(self):
        with self.lock:
            landmarks = self.face_landmarks

        if landmarks is None:
            show_error_popup(self.window, "ไม่พบใบหน้า", "กรุณาหันหน้าเข้ากล้อง")
            return

        if not FaceMLP.check_landmark_quality(landmarks):
            show_error_popup(self.window, "คุณภาพต่ำ", "ใบหน้าไม่ชัดเจน กรุณาลองใหม่")
            return

        features = FaceMLP.extract_features(landmarks, self.use_3d)
        name = self.countdown_expression_name

        if self._countdown_is_new:
            self.expressions[name] = [features.tolist()]
        else:
            if name not in self.expressions:
                self.expressions[name] = []
            self.expressions[name].append(features.tolist())

        self.save_model_to_file()
        self.stats_label.configure(text=self.get_stats_text())
        self.update_expression_list()
        show_success_popup(self.window, f"บันทึก '{name}' สำเร็จ")

    # =========================================================================
    # Dataset import
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
        popup.geometry("450x200")
        popup.transient(self.window)
        popup.grab_set()

        ctk.CTkLabel(popup, text="กำลังประมวลผล...",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10))
        progress = ctk.CTkProgressBar(popup, width=350)
        progress.pack(pady=10, padx=40)
        progress.set(0)
        status = ctk.CTkLabel(popup, text="เริ่มต้น...", font=ctk.CTkFont(size=14))
        status.pack(pady=5)

        def do_import():
            def cb(current, total, class_name):
                try:
                    progress.set(current / total if total > 0 else 0)
                    status.configure(text=f"{class_name}: {current}/{total}")
                except Exception:
                    pass

            result = self.recognition.load_dataset(folder, progress_cb=cb)

            for name, features_list in result.items():
                if name not in self.expressions:
                    self.expressions[name] = []
                self.expressions[name].extend(features_list)

            self.save_model_to_file()

            try:
                total_added = sum(len(v) for v in result.values())
                status.configure(text=f"เสร็จ: {len(result)} ท่า, {total_added} ตัวอย่าง")
                self.stats_label.configure(text=self.get_stats_text())
                self.update_expression_list()
                popup.after(2000, popup.destroy)
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
        if len(self.expressions) < 2:
            show_error_popup(self.window, "ข้อมูลไม่พอ", "ต้องมีอย่างน้อย 2 ท่าทาง")
            return

        popup = ctk.CTkToplevel(self.window)
        popup.title("Training Face MLP")
        popup.geometry("450x300")
        popup.transient(self.window)
        popup.grab_set()

        ctk.CTkLabel(popup, text="กำลัง Training...",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
        pbar = ctk.CTkProgressBar(popup, width=350)
        pbar.pack(pady=10, padx=40); pbar.set(0)
        slabel = ctk.CTkLabel(popup, text="เริ่มต้น...", font=ctk.CTkFont(size=14))
        slabel.pack(pady=5)
        rlabel = ctk.CTkLabel(popup, text="", font=ctk.CTkFont(size=13), justify="left")
        rlabel.pack(pady=10, padx=20)
        close_btn = ctk.CTkButton(popup, text="ปิด", command=popup.destroy,
                                   state="disabled", width=120)
        close_btn.pack(pady=15)

        total_epochs = 300

        def do_train():
            model_data = {"expressions": self.expressions}

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
                        text=f"MLP พร้อม: {len(self.recognition.face_mlp.classes)} ท่า",
                        text_color="#2ecc71")
                close_btn.configure(state="normal")
                pbar.set(1.0)
            except Exception:
                pass

        threading.Thread(target=do_train, daemon=True).start()

    # =========================================================================
    # Expression list
    # =========================================================================

    def get_stats_text(self):
        count = len(self.expressions)
        total = sum(len(v) for v in self.expressions.values())
        return f"ท่าทาง: {count} | ตัวอย่าง: {total}"

    def update_expression_list(self):
        for w in self.expression_list_container.winfo_children():
            w.destroy()

        for name in sorted(self.expressions.keys()):
            samples = self.expressions[name]
            item = ctk.CTkFrame(self.expression_list_container, corner_radius=8, height=40)
            item.pack(padx=5, pady=3, fill="x")
            item.pack_propagate(False)

            is_selected = name == self.selected_expression_name
            fg = "#2c3e50" if is_selected else "transparent"
            item.configure(fg_color=fg)

            ctk.CTkLabel(item, text=f"{name} ({len(samples)})",
                         font=ctk.CTkFont(size=14)).pack(side="left", padx=10)

            ctk.CTkButton(item, text="ลบ", width=50, height=28,
                          fg_color="#e74c3c", hover_color="#c0392b",
                          command=lambda n=name: self.delete_expression(n)).pack(side="right", padx=5)

            item.bind("<Button-1>", lambda e, n=name: self.select_expression(n))

    def select_expression(self, name):
        self.selected_expression_name = name
        self.selected_display.configure(text=name)
        self.selected_frame.pack(padx=10, pady=(0, 15), fill="x")
        self.add_sample_button.configure(state="normal")
        self.update_expression_list()

    def deselect_expression(self):
        self.selected_expression_name = None
        self.selected_frame.pack_forget()
        self.add_sample_button.configure(state="disabled")
        self.update_expression_list()

    def delete_expression(self, name):
        if name in self.expressions:
            del self.expressions[name]
            self.save_model_to_file()
            if self.selected_expression_name == name:
                self.deselect_expression()
            self.stats_label.configure(text=self.get_stats_text())
            self.update_expression_list()

    def clear_expressions(self):
        self.expressions = {}
        self.deselect_expression()
        self.save_model_to_file()
        self.stats_label.configure(text=self.get_stats_text())
        self.update_expression_list()

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
            results = self.face_mesh.process(rgb)

            with self.lock:
                self.frame = frame
                if results.multi_face_landmarks:
                    self.face_landmarks = results.multi_face_landmarks[0]
                    self.face_detected = True
                else:
                    self.face_landmarks = None
                    self.face_detected = False

        cap.release()

    def process_frame(self):
        import time
        while self.running:
            with self.lock:
                frame = self.frame
                landmarks = self.face_landmarks
                detected = self.face_detected

            if frame is None:
                time.sleep(0.03)
                continue

            display = frame.copy()

            # Draw mesh
            if self.show_mesh and detected and landmarks:
                self.mp_drawing.draw_landmarks(
                    display, landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Recognition
            if detected and landmarks and self.expressions and self.show_text_overlay:
                model_data = {"expressions": self.expressions}
                expression, confidence = self.recognition.recognize_from_landmarks(landmarks, model_data)

                if self.use_smoothing:
                    self.smoother.update('face', expression, confidence)
                    expression, confidence = self.smoother.get_smoothed('face')

                if expression and confidence > 0:
                    color = (0, 255, 0) if confidence >= 70 else (0, 165, 255)
                    text = f"{expression} ({confidence:.1f}%)"
                else:
                    color = (128, 128, 128)
                    text = "ไม่รู้จัก"

                display = draw_thai_text(display, text, (20, 20), self.thai_font_video, color)

                try:
                    self.expression_label.configure(text=text if expression else "ไม่รู้จัก")
                except Exception:
                    pass

            # Countdown overlay
            if self.countdown_active and self.countdown_value > 0:
                big_font = load_font(get_asset_path("AmericanoDemo.ttf"), 120)
                display = draw_countdown_overlay(display, self.countdown_value, big_font)

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
        self.face_mesh.close()
        self.window.destroy()

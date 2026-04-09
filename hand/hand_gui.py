"""
Hand Gesture Model Manager GUI
================================
Refactored GUI for hand gesture model creation, training, and comparison.
Uses HandRecognition API and core utilities instead of reimplementing logic.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import time
import os
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog

from hand.hand_recognition import HandRecognition
from hand.hand_mlp import HandMLP
from core.temporal_smoother import TemporalSmoother
from core.utils import (
    draw_thai_text, show_error_popup, show_success_popup,
    display_frame_on_canvas, draw_countdown_overlay,
    load_font, get_asset_path, load_dataset_from_folder
)


class GestureModelManager:
    def __init__(self, window):
        self.window = window
        self.window.title("ระบบจัดการโมเดลท่าทางมือ")
        self.window.geometry("1400x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize MediaPipe (single instance for capture + skeleton drawing)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Load Thai font
        font_path = get_asset_path("AmericanoDemo.ttf")
        self.thai_font = load_font(font_path, 36)
        self.thai_font_video = load_font(font_path, 32)

        # HandRecognition instance for recognition API
        self.recognition = HandRecognition(font_path=font_path, mode="mae", smoothing_window=0)

        # Global variables
        self.current_mode = "create"
        self.current_model_file = None
        self.gestures = {'left': {}, 'right': {}, 'both': {}}

        # For compare mode
        self.model1_file = None
        self.model1_gestures = {'left': {}, 'right': {}, 'both': {}}
        self.model2_file = None
        self.model2_gestures = {'left': {}, 'right': {}, 'both': {}}

        # Selected gesture for adding samples
        self.selected_gesture_name = None
        self.selected_gesture_hand = None

        # Video capture variables
        self.frame = None
        self.hand_landmarks = None
        self.hand_handedness = None
        self.running = True
        self.show_skeleton = True
        self.show_text_overlay = True
        self.canvas_width = 640
        self.canvas_height = 480

        self.lock = threading.Lock()

        # Recognition mode
        self.recognition_mode = "mae"

        # Temporal smoother (GUI-level for display smoothing)
        self.smoother = TemporalSmoother(window_size=5)
        self.use_smoothing = True

        # Countdown state for both-hands capture
        self.countdown_active = False
        self.countdown_value = 0
        self.countdown_gesture_name = None  # None = new gesture, str = name for adding sample

        # Create main layout
        self.create_main_layout()

        # Initialize canvas size after GUI is created
        self.window.update_idletasks()
        self.canvas_width = max(self.canvas.winfo_width(), 640)
        self.canvas_height = max(self.canvas.winfo_height(), 480)

        # Start video threads
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
        # Main container
        self.main_container = ctk.CTkFrame(self.window)
        self.main_container.pack(fill="both", expand=True)

        # Header with mode selector
        self.create_header()

        # Content area
        self.content_frame = ctk.CTkFrame(self.main_container)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Configure content grid
        self.content_frame.grid_columnconfigure(0, weight=7, minsize=400)
        self.content_frame.grid_columnconfigure(1, weight=3, minsize=300)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # Create UI elements
        self.create_video_panel()
        self.create_single_mode_panel()
        self.create_compare_mode_panels()

        # Show initial mode
        self.switch_mode("create")

    def create_header(self):
        header = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        # Title
        title_label = ctk.CTkLabel(
            header,
            text="ระบบจัดการโมเดลท่าทางมือ",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(side="left", padx=30, pady=20)

        # Mode buttons
        button_frame = ctk.CTkFrame(header, fg_color="transparent")
        button_frame.pack(side="right", padx=30, pady=20)

        self.mode_buttons = {}

        btn_create = ctk.CTkButton(
            button_frame,
            text="สร้างโมเดล",
            command=self.create_new_model,
            width=130,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        btn_create.pack(side="left", padx=5)
        self.mode_buttons["create"] = btn_create

        btn_load = ctk.CTkButton(
            button_frame,
            text="เปิดโมเดล",
            command=self.load_model,
            width=130,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        btn_load.pack(side="left", padx=5)
        self.mode_buttons["load"] = btn_load

        btn_compare = ctk.CTkButton(
            button_frame,
            text="เปรียบเทียบ",
            command=lambda: self.switch_mode("compare"),
            width=130,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        btn_compare.pack(side="left", padx=5)
        self.mode_buttons["compare"] = btn_compare

    def create_video_panel(self):
        # Video container - always visible
        video_container = ctk.CTkFrame(self.content_frame, corner_radius=15)
        video_container.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")

        video_container.grid_rowconfigure(1, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        # Mode indicator and model name
        self.mode_info_frame = ctk.CTkFrame(video_container, fg_color="transparent")
        self.mode_info_frame.grid(row=0, column=0, pady=(10, 5), sticky="ew")

        self.mode_label = ctk.CTkLabel(
            self.mode_info_frame,
            text="กล้องสด",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.mode_label.pack(pady=5)

        self.model_name_label = ctk.CTkLabel(
            self.mode_info_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.model_name_label.pack(pady=2)

        self.canvas = ctk.CTkCanvas(
            video_container,
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.canvas_image_id = None

    def create_single_mode_panel(self):
        # Right Panel - Controls for Create/Load mode
        self.single_control_frame = ctk.CTkScrollableFrame(
            self.content_frame,
            corner_radius=15,
            fg_color=("gray90", "gray13")
        )

        # Gesture Display
        gesture_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        gesture_frame.pack(padx=10, pady=(10, 15), fill="x")

        left_container = ctk.CTkFrame(gesture_frame, corner_radius=8)
        left_container.pack(padx=15, pady=(15, 10), fill="x")

        ctk.CTkLabel(
            left_container,
            text="มือซ้าย",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#4a9eff"
        ).pack(pady=(10, 5))

        self.left_gesture_label = ctk.CTkLabel(
            left_container,
            text="ไม่พบมือ",
            font=ctk.CTkFont(size=18)
        )
        self.left_gesture_label.pack(pady=(0, 10))

        right_container = ctk.CTkFrame(gesture_frame, corner_radius=8)
        right_container.pack(padx=15, pady=(10, 10), fill="x")

        ctk.CTkLabel(
            right_container,
            text="มือขวา",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#4a9eff"
        ).pack(pady=(10, 5))

        self.right_gesture_label = ctk.CTkLabel(
            right_container,
            text="ไม่พบมือ",
            font=ctk.CTkFont(size=18)
        )
        self.right_gesture_label.pack(pady=(0, 10))

        both_container = ctk.CTkFrame(gesture_frame, corner_radius=8)
        both_container.pack(padx=15, pady=(10, 15), fill="x")

        ctk.CTkLabel(
            both_container,
            text="2 มือ",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e67e22"
        ).pack(pady=(10, 5))

        self.both_gesture_label = ctk.CTkLabel(
            both_container,
            text="-",
            font=ctk.CTkFont(size=18)
        )
        self.both_gesture_label.pack(pady=(0, 10))

        # Settings
        settings_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        settings_frame.pack(padx=10, pady=(0, 15), fill="x")

        ctk.CTkLabel(
            settings_frame,
            text="การตั้งค่า",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))

        self.skeleton_switch = ctk.CTkSwitch(
            settings_frame,
            text="แสดงโครงกระดูกมือ",
            font=ctk.CTkFont(size=14),
            command=self.toggle_skeleton
        )
        self.skeleton_switch.pack(padx=15, pady=8)
        self.skeleton_switch.select()

        self.text_switch = ctk.CTkSwitch(
            settings_frame,
            text="แสดงข้อความบนวิดีโอ",
            font=ctk.CTkFont(size=14),
            command=self.toggle_text
        )
        self.text_switch.pack(padx=15, pady=(8, 8))
        self.text_switch.select()

        self.smoothing_switch = ctk.CTkSwitch(
            settings_frame,
            text="Temporal Smoothing",
            font=ctk.CTkFont(size=14),
            command=self.toggle_smoothing
        )
        self.smoothing_switch.pack(padx=15, pady=(8, 15))
        self.smoothing_switch.select()

        # MLP Controls
        mlp_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        mlp_frame.pack(padx=10, pady=(0, 15), fill="x")

        ctk.CTkLabel(
            mlp_frame,
            text="MLP Neural Network",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))

        # Mode selector
        mode_container = ctk.CTkFrame(mlp_frame, fg_color="transparent")
        mode_container.pack(padx=15, pady=(0, 8), fill="x")

        ctk.CTkLabel(
            mode_container,
            text="โหมด:",
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=(0, 10))

        self.mode_var = ctk.StringVar(value="MAE (ดั้งเดิม)")
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_container,
            values=["MAE (ดั้งเดิม)", "MLP (Neural)"],
            variable=self.mode_var,
            command=self.switch_recognition_mode,
            font=ctk.CTkFont(size=12)
        )
        self.mode_selector.pack(side="left", fill="x", expand=True)

        self.train_mlp_button = ctk.CTkButton(
            mlp_frame,
            text="Train MLP จากข้อมูลปัจจุบัน",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.train_mlp_model,
            fg_color="#8e44ad",
            hover_color="#7d3c98"
        )
        self.train_mlp_button.pack(padx=15, pady=(0, 8), fill="x")

        self.mlp_status_label = ctk.CTkLabel(
            mlp_frame,
            text="ยังไม่ได้ train MLP",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.mlp_status_label.pack(padx=15, pady=(0, 15))

        # Action Buttons
        button_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        button_frame.pack(padx=10, pady=(0, 15), fill="x")

        ctk.CTkLabel(
            button_frame,
            text="การดำเนินการ",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))

        self.save_new_button = ctk.CTkButton(
            button_frame,
            text="บันทึกท่าทางใหม่ (มือเดียว)",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.save_new_gesture,
            fg_color="#3498db",
            hover_color="#2980b9"
        )
        self.save_new_button.pack(padx=15, pady=(0, 10), fill="x")

        self.save_both_button = ctk.CTkButton(
            button_frame,
            text="บันทึกท่าทาง 2 มือ (นับถอยหลัง)",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.save_both_hands_gesture,
            fg_color="#e67e22",
            hover_color="#d35400"
        )
        self.save_both_button.pack(padx=15, pady=(0, 10), fill="x")

        self.import_dataset_button = ctk.CTkButton(
            button_frame,
            text="นำเข้าจาก Dataset",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.load_dataset_folder,
            fg_color="#1abc9c",
            hover_color="#16a085"
        )
        self.import_dataset_button.pack(padx=15, pady=(0, 10), fill="x")

        self.clear_button = ctk.CTkButton(
            button_frame,
            text="ลบท่าทางทั้งหมด",
            font=ctk.CTkFont(size=15),
            height=40,
            command=self.clear_gestures,
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.clear_button.pack(padx=15, pady=(0, 15), fill="x")

        # Statistics
        stats_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        stats_frame.pack(padx=10, pady=(0, 15), fill="x")

        ctk.CTkLabel(
            stats_frame,
            text="สถิติ",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))

        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text=self.get_stats_text(),
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        self.stats_label.pack(padx=15, pady=(0, 15))

        # Selected Gesture Indicator
        self.selected_indicator_frame = ctk.CTkFrame(
            self.single_control_frame, corner_radius=10, fg_color="#2c3e50"
        )

        ctk.CTkLabel(
            self.selected_indicator_frame,
            text="ท่าทางที่เลือก",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f39c12"
        ).pack(pady=(12, 5))

        self.selected_gesture_display = ctk.CTkLabel(
            self.selected_indicator_frame,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#ecf0f1"
        )
        self.selected_gesture_display.pack(pady=(0, 5))

        self.deselect_button = ctk.CTkButton(
            self.selected_indicator_frame,
            text="ยกเลิกการเลือก",
            command=self.deselect_gesture,
            font=ctk.CTkFont(size=13),
            height=35,
            fg_color="#e67e22",
            hover_color="#d35400"
        )
        self.deselect_button.pack(padx=15, pady=(5, 12), fill="x")

        # Gesture List
        list_frame = ctk.CTkFrame(self.single_control_frame, corner_radius=10)
        list_frame.pack(padx=10, pady=(0, 15), fill="both", expand=True)

        header_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(15, 10), padx=15)

        ctk.CTkLabel(
            header_frame,
            text="ท่าทางที่บันทึกไว้",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            header_frame,
            text="(คลิกเพื่อเลือก)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(side="left", padx=(10, 0))

        self.gesture_list_container = ctk.CTkScrollableFrame(
            list_frame,
            height=200,
            fg_color=("gray85", "gray20")
        )
        self.gesture_list_container.pack(padx=15, pady=(0, 10), fill="both", expand=True)

        self.list_frame = list_frame

        self.add_sample_button = ctk.CTkButton(
            list_frame,
            text="เพิ่มตัวอย่างท่าทาง",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=45,
            command=self.add_gesture_sample,
            fg_color="#2ecc71",
            hover_color="#27ae60",
            state="disabled"
        )
        self.add_sample_button.pack(padx=15, pady=(0, 15), fill="x")

        self.update_gesture_list()

    def create_compare_mode_panels(self):
        # Container for both model panels
        self.compare_container = ctk.CTkFrame(self.content_frame, corner_radius=15)

        self.compare_container.grid_columnconfigure(0, weight=1)
        self.compare_container.grid_columnconfigure(1, weight=1)
        self.compare_container.grid_rowconfigure(0, weight=1)

        # Model 1 Panel
        model1_frame = ctk.CTkFrame(self.compare_container, corner_radius=15)
        model1_frame.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        self.create_compare_panel(model1_frame, 1)

        # Model 2 Panel
        model2_frame = ctk.CTkFrame(self.compare_container, corner_radius=15)
        model2_frame.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        self.create_compare_panel(model2_frame, 2)

    def create_compare_panel(self, parent, model_num):
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(parent, corner_radius=10, fg_color="#2c3e50")
        header.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")

        ctk.CTkLabel(
            header,
            text=f"โมเดล {model_num}",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=10)

        # Load button
        load_btn = ctk.CTkButton(
            parent,
            text="โหลดโมเดล",
            command=lambda: self.load_compare_model(model_num),
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#9b59b6",
            hover_color="#8e44ad"
        )
        load_btn.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")

        # Info display
        info_frame = ctk.CTkFrame(parent, corner_radius=10)
        info_frame.grid(row=2, column=0, padx=15, pady=(0, 10), sticky="nsew")

        if model_num == 1:
            self.model1_info = ctk.CTkTextbox(
                info_frame,
                font=ctk.CTkFont(size=13),
                wrap="word"
            )
            self.model1_info.pack(padx=10, pady=10, fill="both", expand=True)
            self.model1_info.insert("1.0", "ยังไม่ได้โหลดโมเดล")
        else:
            self.model2_info = ctk.CTkTextbox(
                info_frame,
                font=ctk.CTkFont(size=13),
                wrap="word"
            )
            self.model2_info.pack(padx=10, pady=10, fill="both", expand=True)
            self.model2_info.insert("1.0", "ยังไม่ได้โหลดโมเดล")

        # Gesture display
        gesture_frame = ctk.CTkFrame(parent, corner_radius=10)
        gesture_frame.grid(row=3, column=0, padx=15, pady=(0, 15), sticky="ew")

        ctk.CTkLabel(
            gesture_frame,
            text="มือซ้าย",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4a9eff"
        ).pack(pady=(10, 5))

        if model_num == 1:
            self.model1_left_label = ctk.CTkLabel(
                gesture_frame,
                text="ไม่พบมือ",
                font=ctk.CTkFont(size=16)
            )
            self.model1_left_label.pack(pady=(0, 5))
        else:
            self.model2_left_label = ctk.CTkLabel(
                gesture_frame,
                text="ไม่พบมือ",
                font=ctk.CTkFont(size=16)
            )
            self.model2_left_label.pack(pady=(0, 5))

        ctk.CTkLabel(
            gesture_frame,
            text="มือขวา",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4a9eff"
        ).pack(pady=(10, 5))

        if model_num == 1:
            self.model1_right_label = ctk.CTkLabel(
                gesture_frame,
                text="ไม่พบมือ",
                font=ctk.CTkFont(size=16)
            )
            self.model1_right_label.pack(pady=(0, 10))
        else:
            self.model2_right_label = ctk.CTkLabel(
                gesture_frame,
                text="ไม่พบมือ",
                font=ctk.CTkFont(size=16)
            )
            self.model2_right_label.pack(pady=(0, 10))

    # =========================================================================
    # Mode switching
    # =========================================================================

    def switch_mode(self, mode):
        self.current_mode = mode

        # Hide all right panels
        self.single_control_frame.grid_forget()
        self.compare_container.grid_forget()

        # Show appropriate panel
        if mode == "compare":
            self.compare_container.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        else:
            self.single_control_frame.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")

        # Reset canvas
        self.window.update_idletasks()
        self.canvas_image_id = None

    # =========================================================================
    # Model I/O
    # =========================================================================

    def create_new_model(self):
        dialog = ctk.CTkInputDialog(
            text="ชื่อไฟล์โมเดล (ไม่ต้องใส่ .json):",
            title="สร้างโมเดลใหม่"
        )
        filename = dialog.get_input()

        if filename:
            if not filename.endswith('.json'):
                filename += '.json'

            if os.path.exists(filename):
                show_error_popup(self.window, "ข้อผิดพลาด", "ไฟล์นี้มีอยู่แล้ว")
                return

            self.current_model_file = filename
            self.gestures = {'left': {}, 'right': {}, 'both': {}}

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.gestures, f, ensure_ascii=False, indent=2)

            self.model_name_label.configure(text=f"{filename}")
            self.stats_label.configure(text=self.get_stats_text())
            self.update_gesture_list()

            self.switch_mode("create")

            show_success_popup(self.window, "สร้างโมเดลสำเร็จ")

    def load_model(self):
        filename = filedialog.askopenfilename(
            title="เลือกไฟล์โมเดล",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.gestures = json.load(f)

                # Ensure 'both' key exists for older models
                if 'both' not in self.gestures:
                    self.gestures['both'] = {}

                self.current_model_file = filename
                self.model_name_label.configure(text=f"{os.path.basename(filename)}")
                self.stats_label.configure(text=self.get_stats_text())
                self.update_gesture_list()

                # Auto-load MLP models via recognition API
                mlp_loaded = []
                if self.recognition.load_mlp(filename):
                    if self.recognition.mlp_left.is_trained:
                        mlp_loaded.append(f"ซ้าย: {len(self.recognition.mlp_left.classes)} ท่า")
                    if self.recognition.mlp_right.is_trained:
                        mlp_loaded.append(f"ขวา: {len(self.recognition.mlp_right.classes)} ท่า")
                    if self.recognition.mlp_both.is_trained:
                        mlp_loaded.append(f"2มือ: {len(self.recognition.mlp_both.classes)} ท่า")

                if mlp_loaded:
                    self.mlp_status_label.configure(
                        text="MLP พร้อมใช้งาน | " + " | ".join(mlp_loaded),
                        text_color="#2ecc71"
                    )
                else:
                    self.mlp_status_label.configure(
                        text="ยังไม่ได้ train MLP",
                        text_color="gray"
                    )

                self.switch_mode("load")

                success_text = "โหลดโมเดลสำเร็จ"
                if mlp_loaded:
                    success_text += "\n(พบ MLP model)"

                show_success_popup(self.window, success_text, auto_close_ms=1500)

            except Exception as e:
                show_error_popup(self.window, "ไม่สามารถโหลดไฟล์ได้", str(e))

    def load_compare_model(self, model_num):
        filename = filedialog.askopenfilename(
            title=f"เลือกโมเดล {model_num}",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    gestures = json.load(f)

                # Ensure 'both' key exists for older models
                if 'both' not in gestures:
                    gestures['both'] = {}

                if model_num == 1:
                    self.model1_file = filename
                    self.model1_gestures = gestures
                    self.update_model_info(1)
                else:
                    self.model2_file = filename
                    self.model2_gestures = gestures
                    self.update_model_info(2)

            except Exception as e:
                show_error_popup(self.window, "ไม่สามารถโหลดไฟล์ได้", str(e))

    def update_model_info(self, model_num):
        if model_num == 1:
            gestures = self.model1_gestures
            filename = self.model1_file
            textbox = self.model1_info
        else:
            gestures = self.model2_gestures
            filename = self.model2_file
            textbox = self.model2_info

        left_count = len(gestures.get('left', {}))
        right_count = len(gestures.get('right', {}))
        both_count = len(gestures.get('both', {}))
        total = left_count + right_count + both_count

        left_variations = sum(len(v) for v in gestures.get('left', {}).values())
        right_variations = sum(len(v) for v in gestures.get('right', {}).values())
        both_variations = sum(len(v) for v in gestures.get('both', {}).values())

        info_text = f"""ไฟล์: {os.path.basename(filename)}

สถิติ:
ท่าทางทั้งหมด: {total} ท่า
มือซ้าย: {left_count} ท่า ({left_variations} แบบ)
มือขวา: {right_count} ท่า ({right_variations} แบบ)
2 มือ: {both_count} ท่า ({both_variations} แบบ)

รายการท่าทาง:
"""

        if gestures.get('left', {}):
            info_text += "\nมือซ้าย:\n"
            for name, variations in gestures['left'].items():
                info_text += f"   • {name} ({len(variations)} แบบ)\n"

        if gestures.get('right', {}):
            info_text += "\nมือขวา:\n"
            for name, variations in gestures['right'].items():
                info_text += f"   • {name} ({len(variations)} แบบ)\n"

        if gestures.get('both', {}):
            info_text += "\n2 มือ:\n"
            for name, variations in gestures['both'].items():
                info_text += f"   • {name} ({len(variations)} แบบ)\n"

        textbox.delete("1.0", "end")
        textbox.insert("1.0", info_text)

    # =========================================================================
    # Settings
    # =========================================================================

    def update_canvas_size(self):
        try:
            new_width = self.canvas.winfo_width()
            new_height = self.canvas.winfo_height()
            if new_width > 50 and new_height > 50:
                self.canvas_width = new_width
                self.canvas_height = new_height
        except:
            pass

    def toggle_skeleton(self):
        self.show_skeleton = self.skeleton_switch.get()

    def toggle_text(self):
        self.show_text_overlay = self.text_switch.get()

    def toggle_smoothing(self):
        self.use_smoothing = self.smoothing_switch.get()
        if not self.use_smoothing:
            self.smoother.reset()

    def switch_recognition_mode(self, value):
        if "MLP" in value:
            if (not self.recognition.mlp_left.is_trained
                    and not self.recognition.mlp_right.is_trained
                    and not self.recognition.mlp_both.is_trained):
                show_error_popup(
                    self.window,
                    "ยังไม่ได้ Train MLP",
                    "กรุณา Train MLP ก่อนใช้โหมดนี้\nกดปุ่ม 'Train MLP จากข้อมูลปัจจุบัน'"
                )
                self.mode_var.set("MAE (ดั้งเดิม)")
                return
            self.recognition_mode = "mlp"
        else:
            self.recognition_mode = "mae"
        self.recognition.set_mode(self.recognition_mode)
        self.smoother.reset()

    # =========================================================================
    # MLP Training
    # =========================================================================

    def train_mlp_model(self):
        if not self.current_model_file:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกโมเดล",
                "กรุณาสร้างหรือเปิดโมเดลก่อน"
            )
            return

        left_count = len(self.gestures.get('left', {}))
        right_count = len(self.gestures.get('right', {}))
        both_count = len(self.gestures.get('both', {}))

        if left_count < 2 and right_count < 2 and both_count < 2:
            show_error_popup(
                self.window,
                "ข้อมูลไม่เพียงพอ",
                "ต้องมีอย่างน้อย 2 ท่าทาง\nในมือซ้าย มือขวา หรือ 2 มือ"
            )
            return

        # Show training progress popup
        progress_popup = ctk.CTkToplevel(self.window)
        progress_popup.title("Training MLP")
        progress_popup.geometry("450x350")
        progress_popup.transient(self.window)
        progress_popup.grab_set()

        progress_popup.update_idletasks()
        x = (progress_popup.winfo_screenwidth() // 2) - 225
        y = (progress_popup.winfo_screenheight() // 2) - 175
        progress_popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            progress_popup,
            text="กำลัง Training MLP...",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(20, 10))

        progress_bar = ctk.CTkProgressBar(progress_popup, width=350)
        progress_bar.pack(pady=10, padx=40)
        progress_bar.set(0)

        status_label = ctk.CTkLabel(
            progress_popup,
            text="เริ่มต้น...",
            font=ctk.CTkFont(size=14)
        )
        status_label.pack(pady=5)

        result_label = ctk.CTkLabel(
            progress_popup,
            text="",
            font=ctk.CTkFont(size=13),
            justify="left"
        )
        result_label.pack(pady=10, padx=20)

        close_btn = ctk.CTkButton(
            progress_popup,
            text="ปิด",
            command=progress_popup.destroy,
            state="disabled",
            width=120
        )
        close_btn.pack(pady=15)

        total_epochs = 200

        def do_train():
            results_text = ""
            hand_names = {'left': 'มือซ้าย', 'right': 'มือขวา', 'both': '2 มือ'}

            def progress_callback(hand_type, epoch, train_loss, train_acc, val_loss, val_acc):
                hn = hand_names.get(hand_type, hand_type)
                try:
                    progress_bar.set((epoch + 1) / total_epochs)
                    status_label.configure(
                        text=f"{hn}: Epoch {epoch+1}/{total_epochs} | "
                             f"Loss: {train_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%"
                    )
                except:
                    pass

            # Use recognition API to train all hand types
            results = self.recognition.train_mlp(
                self.gestures, epochs=total_epochs, verbose=False,
                progress_callback=progress_callback
            )

            for hand_type in ['left', 'right', 'both']:
                hand_name = hand_names.get(hand_type, hand_type)
                stats = results.get(hand_type)
                if stats is None:
                    gesture_count = len(self.gestures.get(hand_type, {}))
                    results_text += f"{hand_name}: ข้ามไป (มี {gesture_count} ท่า)\n"
                else:
                    val_acc = stats.get('final_val_accuracy', stats.get('final_accuracy', 0))
                    results_text += (f"{hand_name}: Train {stats['final_accuracy']:.1f}% | "
                                     f"Val {val_acc:.1f}% | "
                                     f"Loss {stats['final_loss']:.4f}\n")

            # Save MLP alongside the gesture model
            if self.current_model_file:
                self.recognition.save_mlp(self.current_model_file)

            try:
                status_label.configure(text="Training เสร็จสิ้น!")
                result_label.configure(text=results_text)
                close_btn.configure(state="normal")
                progress_bar.set(1.0)

                mlp_status = []
                if self.recognition.mlp_left.is_trained:
                    mlp_status.append(f"ซ้าย: {len(self.recognition.mlp_left.classes)} ท่า")
                if self.recognition.mlp_right.is_trained:
                    mlp_status.append(f"ขวา: {len(self.recognition.mlp_right.classes)} ท่า")
                if self.recognition.mlp_both.is_trained:
                    mlp_status.append(f"2มือ: {len(self.recognition.mlp_both.classes)} ท่า")
                self.mlp_status_label.configure(
                    text="MLP พร้อมใช้งาน | " + " | ".join(mlp_status),
                    text_color="#2ecc71"
                )
            except:
                pass

        threading.Thread(target=do_train, daemon=True).start()

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats_text(self):
        left_count = len(self.gestures.get('left', {}))
        right_count = len(self.gestures.get('right', {}))
        both_count = len(self.gestures.get('both', {}))
        total = left_count + right_count + both_count

        left_samples = sum(len(v) for v in self.gestures.get('left', {}).values())
        right_samples = sum(len(v) for v in self.gestures.get('right', {}).values())
        both_samples = sum(len(v) for v in self.gestures.get('both', {}).values())

        return f"""ท่าทางทั้งหมด: {total} ท่า
มือซ้าย: {left_count} ท่า ({left_samples} ตัวอย่าง)
มือขวา: {right_count} ท่า ({right_samples} ตัวอย่าง)
2 มือ: {both_count} ท่า ({both_samples} ตัวอย่าง)"""

    # =========================================================================
    # Gesture List
    # =========================================================================

    def update_gesture_list(self):
        for widget in self.gesture_list_container.winfo_children():
            widget.destroy()

        has_any = (any(self.gestures.get('left', {})) or
                   any(self.gestures.get('right', {})) or
                   any(self.gestures.get('both', {})))

        if not has_any:
            no_data_label = ctk.CTkLabel(
                self.gesture_list_container,
                text="ยังไม่มีท่าทางที่บันทึกไว้",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            no_data_label.pack(pady=20)
        else:
            if self.gestures.get('left', {}):
                left_header = ctk.CTkLabel(
                    self.gesture_list_container,
                    text="มือซ้าย",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="#4a9eff",
                    anchor="w"
                )
                left_header.pack(fill="x", pady=(5, 5), padx=5)

                for name, samples in self.gestures['left'].items():
                    self.create_gesture_item(name, 'left', len(samples))

            if self.gestures.get('right', {}):
                right_header = ctk.CTkLabel(
                    self.gesture_list_container,
                    text="มือขวา",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="#4a9eff",
                    anchor="w"
                )
                right_header.pack(fill="x", pady=(15, 5), padx=5)

                for name, samples in self.gestures['right'].items():
                    self.create_gesture_item(name, 'right', len(samples))

            if self.gestures.get('both', {}):
                both_header = ctk.CTkLabel(
                    self.gesture_list_container,
                    text="2 มือ",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="#e67e22",
                    anchor="w"
                )
                both_header.pack(fill="x", pady=(15, 5), padx=5)

                for name, samples in self.gestures['both'].items():
                    self.create_gesture_item(name, 'both', len(samples))

        self.update_selected_indicator()

    def create_gesture_item(self, name, hand_type, sample_count):
        is_selected = (self.selected_gesture_name == name and
                       self.selected_gesture_hand == hand_type)

        item_frame = ctk.CTkFrame(
            self.gesture_list_container,
            corner_radius=8,
            fg_color=("#16a085" if is_selected else "#34495e"),
            cursor="hand2"
        )
        item_frame.pack(fill="x", pady=3, padx=5)

        item_frame.bind("<Button-1>", lambda e: self.select_gesture(name, hand_type))

        content_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=8)

        name_label = ctk.CTkLabel(
            content_frame,
            text=f"{name}",
            font=ctk.CTkFont(size=14, weight="bold" if is_selected else "normal"),
            anchor="w"
        )
        name_label.pack(side="left", fill="x", expand=True)
        name_label.bind("<Button-1>", lambda e: self.select_gesture(name, hand_type))

        badge = ctk.CTkLabel(
            content_frame,
            text=f"{sample_count} ตัวอย่าง",
            font=ctk.CTkFont(size=11),
            fg_color=("#e67e22" if is_selected else "#7f8c8d"),
            corner_radius=10,
            width=80,
            height=22
        )
        badge.pack(side="right", padx=(5, 0))
        badge.bind("<Button-1>", lambda e: self.select_gesture(name, hand_type))

        if is_selected:
            indicator = ctk.CTkLabel(
                content_frame,
                text="✓",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#ecf0f1",
                width=25
            )
            indicator.pack(side="right", padx=(0, 5))
            indicator.bind("<Button-1>", lambda e: self.select_gesture(name, hand_type))

    def select_gesture(self, name, hand_type):
        self.selected_gesture_name = name
        self.selected_gesture_hand = hand_type
        self.update_gesture_list()
        self.update_selected_indicator()

    def deselect_gesture(self):
        self.selected_gesture_name = None
        self.selected_gesture_hand = None
        self.update_gesture_list()
        self.update_selected_indicator()

    def update_selected_indicator(self):
        if self.selected_gesture_name and self.selected_gesture_hand:
            hand_names = {'left': 'มือซ้าย', 'right': 'มือขวา', 'both': '2 มือ'}
            hand_text = hand_names.get(self.selected_gesture_hand, self.selected_gesture_hand)
            self.selected_gesture_display.configure(
                text=f"{self.selected_gesture_name} ({hand_text})"
            )
            self.selected_indicator_frame.pack(padx=10, pady=(0, 15), fill="x", before=self.list_frame)
            self.add_sample_button.configure(state="normal")
        else:
            self.selected_indicator_frame.pack_forget()
            self.add_sample_button.configure(state="disabled")

    # =========================================================================
    # Clear gestures
    # =========================================================================

    def clear_gestures(self):
        if not self.current_model_file:
            show_error_popup(self.window, "ข้อผิดพลาด", "ยังไม่ได้เลือกโมเดล")
            return

        popup = ctk.CTkToplevel(self.window)
        popup.title("ยืนยันการลบ")
        popup.geometry("380x200")
        popup.transient(self.window)
        popup.grab_set()

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="คำเตือน",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            popup,
            text="คุณแน่ใจหรือไม่ที่จะลบท่าทาง\nที่บันทึกไว้ทั้งหมด?",
            font=ctk.CTkFont(size=15)
        ).pack(pady=10)

        button_frame = ctk.CTkFrame(popup, fg_color="transparent")
        button_frame.pack(pady=20)

        def confirm():
            self.gestures = {'left': {}, 'right': {}, 'both': {}}
            with open(self.current_model_file, 'w', encoding='utf-8') as f:
                json.dump(self.gestures, f, ensure_ascii=False)
            self.stats_label.configure(text=self.get_stats_text())
            self.deselect_gesture()
            self.update_gesture_list()
            popup.destroy()

        ctk.CTkButton(
            button_frame,
            text="ใช่, ลบเลย",
            command=confirm,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=130,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="ยกเลิก",
            command=popup.destroy,
            width=130,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

    # =========================================================================
    # Gesture Capture — Single Hand
    # =========================================================================

    def crop_hand(self, image, landmarks):
        h, w, _ = image.shape
        x_min = int(min(landmark.x for landmark in landmarks.landmark) * w)
        x_max = int(max(landmark.x for landmark in landmarks.landmark) * w)
        y_min = int(min(landmark.y for landmark in landmarks.landmark) * h)
        y_max = int(max(landmark.y for landmark in landmarks.landmark) * h)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        return image[y_min:y_max, x_min:x_max]

    def save_new_gesture(self):
        if not self.current_model_file:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกโมเดล",
                "กรุณาสร้างหรือเปิดโมเดล\nก่อนบันทึกท่าทาง"
            )
            return

        with self.lock:
            if not self.hand_landmarks or not self.hand_handedness:
                show_error_popup(
                    self.window,
                    "ไม่พบมือ",
                    "กรุณาแสดงมือให้กล้องเห็น\nก่อนบันทึกท่าทาง"
                )
                return

            hand_label = self.hand_handedness[0].classification[0].label
            hand_type = 'left' if hand_label == 'Left' else 'right'
            cropped_frame = self.crop_hand(self.frame, self.hand_landmarks[0])
            landmarks = self.hand_landmarks[0]

            self.show_new_gesture_popup(cropped_frame, landmarks, hand_type)

    def add_gesture_sample(self):
        if not self.current_model_file:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกโมเดล",
                "กรุณาสร้างหรือเปิดโมเดล\nก่อนบันทึกท่าทาง"
            )
            return

        if not self.selected_gesture_name or not self.selected_gesture_hand:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกท่าทาง",
                "กรุณาเลือกท่าทางจากรายการ\nก่อนเพิ่มตัวอย่าง"
            )
            return

        # For both-hands gesture, use countdown then capture
        if self.selected_gesture_hand == 'both':
            self.countdown_gesture_name = self.selected_gesture_name
            self.start_countdown_capture(is_new=False)
            return

        with self.lock:
            if not self.hand_landmarks or not self.hand_handedness:
                show_error_popup(
                    self.window,
                    "ไม่พบมือ",
                    "กรุณาแสดงมือให้กล้องเห็น\nก่อนบันทึกท่าทาง"
                )
                return

            hand_label = self.hand_handedness[0].classification[0].label
            detected_hand_type = 'left' if hand_label == 'Left' else 'right'

            if detected_hand_type != self.selected_gesture_hand:
                hand_name = "ซ้าย" if self.selected_gesture_hand == 'left' else "ขวา"
                show_error_popup(
                    self.window,
                    "มือไม่ตรงกัน",
                    f"ท่าทางที่เลือกเป็นมือ{hand_name}\nกรุณาแสดงมือ{hand_name}ให้กล้องเห็น"
                )
                return

            cropped_frame = self.crop_hand(self.frame, self.hand_landmarks[0])
            landmarks = self.hand_landmarks[0]

            self.quick_add_sample(cropped_frame, landmarks)

    def show_new_gesture_popup(self, cropped_frame, landmarks, hand_type):
        popup = ctk.CTkToplevel(self.window)
        popup.title("บันทึกท่าทางใหม่")
        popup.minsize(450, 600)
        popup.transient(self.window)
        popup.grab_set()

        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(popup)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - 225
        y = (popup.winfo_screenheight() // 2) - 300
        popup.geometry(f"450x600+{x}+{y}")

        hand_text = "มือซ้าย" if hand_type == 'left' else "มือขวา"

        ctk.CTkLabel(
            main_frame,
            text=f"บันทึกท่าทางใหม่ {hand_text}",
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, pady=(10, 10))

        image_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        image_frame.grid(row=1, column=0, pady=10, sticky="ew")

        image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        max_size = 350
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)

        image_label = ctk.CTkLabel(image_frame, image=ctk_image, text="")
        image_label.pack(padx=10, pady=10)

        ctk.CTkLabel(
            main_frame,
            text="ชื่อท่าทาง:",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=2, column=0, pady=(20, 5))

        name_entry = ctk.CTkEntry(
            main_frame,
            height=45,
            font=ctk.CTkFont(size=15),
            placeholder_text="ใส่ชื่อท่าทาง..."
        )
        name_entry.grid(row=3, column=0, pady=5, padx=20, sticky="ew")
        name_entry.focus()

        def save():
            gesture_name = name_entry.get().strip()
            if not gesture_name:
                return

            if gesture_name in self.gestures[hand_type]:
                popup.destroy()
                show_error_popup(
                    self.window,
                    "ชื่อนี้มีอยู่แล้ว",
                    f"มีท่าทาง '{gesture_name}' อยู่แล้ว\nกรุณาใช้ปุ่ม 'เพิ่มตัวอย่าง'\nหรือตั้งชื่อใหม่"
                )
                return

            # Store 225-dim features (210 distances + 15 real angles)
            features = HandMLP.extract_single_hand_features(landmarks).tolist()
            self.gestures[hand_type][gesture_name] = [features]

            with open(self.current_model_file, 'w', encoding='utf-8') as f:
                json.dump(self.gestures, f, ensure_ascii=False, indent=2)

            self.stats_label.configure(text=self.get_stats_text())
            self.update_gesture_list()

            popup.destroy()

            show_success_popup(self.window, f"บันทึกท่าทาง '{gesture_name}' เรียบร้อย", auto_close_ms=2000)

        def cancel():
            popup.destroy()

        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.grid(row=4, column=0, pady=25)

        ctk.CTkButton(
            button_frame,
            text="ยืนยัน",
            command=save,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2ecc71",
            hover_color="#27ae60",
            width=160,
            height=50
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            button_frame,
            text="ยกเลิก",
            command=cancel,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#95a5a6",
            hover_color="#7f8c8d",
            width=160,
            height=50
        ).pack(side="left", padx=8)

        popup.bind('<Return>', lambda e: save())

    def quick_add_sample(self, cropped_frame, landmarks):
        gesture_name = self.selected_gesture_name
        hand_type = self.selected_gesture_hand

        # Store 225-dim features (210 distances + 15 real angles)
        features = HandMLP.extract_single_hand_features(landmarks).tolist()
        self.gestures[hand_type][gesture_name].append(features)

        with open(self.current_model_file, 'w', encoding='utf-8') as f:
            json.dump(self.gestures, f, ensure_ascii=False, indent=2)

        self.stats_label.configure(text=self.get_stats_text())
        self.update_gesture_list()

    # =========================================================================
    # Gesture Capture — Both Hands (countdown)
    # =========================================================================

    def save_both_hands_gesture(self):
        """Save a both-hands gesture -- ask for name then start countdown."""
        if not self.current_model_file:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกโมเดล",
                "กรุณาสร้างหรือเปิดโมเดลก่อน"
            )
            return

        dialog = ctk.CTkInputDialog(
            text="ชื่อท่าทาง 2 มือ:",
            title="บันทึกท่าทาง 2 มือ"
        )
        gesture_name = dialog.get_input()

        if not gesture_name or not gesture_name.strip():
            return

        gesture_name = gesture_name.strip()

        if gesture_name in self.gestures.get('both', {}):
            show_error_popup(
                self.window,
                "ชื่อนี้มีอยู่แล้ว",
                f"มีท่าทาง '{gesture_name}' อยู่แล้ว\nกรุณาใช้ปุ่ม 'เพิ่มตัวอย่าง' หรือตั้งชื่อใหม่"
            )
            return

        self.countdown_gesture_name = gesture_name
        self.start_countdown_capture(is_new=True)

    def start_countdown_capture(self, is_new: bool = True):
        """Start countdown 3-2-1 then capture both-hands gesture."""
        self.countdown_active = True
        self.countdown_value = 3
        self._countdown_is_new = is_new
        self._countdown_tick()

    def _countdown_tick(self):
        """Callback for each countdown second."""
        if self.countdown_value > 0:
            self.countdown_value -= 1
            if self.countdown_value > 0:
                self.window.after(1000, self._countdown_tick)
            else:
                # countdown = 0, capture now!
                self.window.after(1000, self._capture_both_hands)
        else:
            self._capture_both_hands()

    def _capture_both_hands(self):
        """Capture both-hands gesture after countdown ends."""
        self.countdown_active = False
        self.countdown_value = 0

        with self.lock:
            if not self.hand_landmarks or not self.hand_handedness:
                show_error_popup(
                    self.window,
                    "ไม่พบมือ",
                    "ไม่พบมือทั้ง 2 ข้าง\nกรุณาลองใหม่"
                )
                return

            # Find left and right landmarks
            left_landmarks = None
            right_landmarks = None
            for landmarks, handedness in zip(self.hand_landmarks, self.hand_handedness):
                hand_label = handedness.classification[0].label
                if hand_label == 'Left':
                    left_landmarks = landmarks
                else:
                    right_landmarks = landmarks

            if left_landmarks is None or right_landmarks is None:
                show_error_popup(
                    self.window,
                    "ต้องเห็นมือทั้ง 2 ข้าง",
                    "กรุณาแสดงมือซ้ายและขวา\nให้กล้องเห็นพร้อมกัน"
                )
                return

            # Extract 861-dim feature vector (normalized) via HandMLP
            features = HandMLP.extract_both_hands_features(
                left_landmarks, right_landmarks
            ).tolist()

        gesture_name = self.countdown_gesture_name

        if self._countdown_is_new:
            # New gesture
            if 'both' not in self.gestures:
                self.gestures['both'] = {}
            self.gestures['both'][gesture_name] = [features]
        else:
            # Add sample to existing gesture
            if gesture_name in self.gestures.get('both', {}):
                self.gestures['both'][gesture_name].append(features)
            else:
                self.gestures['both'][gesture_name] = [features]

        # Save to file
        with open(self.current_model_file, 'w', encoding='utf-8') as f:
            json.dump(self.gestures, f, ensure_ascii=False, indent=2)

        self.stats_label.configure(text=self.get_stats_text())
        self.update_gesture_list()

        action_text = "บันทึก" if self._countdown_is_new else "เพิ่มตัวอย่าง"
        show_success_popup(
            self.window,
            f"{action_text}ท่าทาง 2 มือสำเร็จ\n'{gesture_name}'",
            auto_close_ms=1500
        )

    # =========================================================================
    # Dataset Import
    # =========================================================================

    def load_dataset_folder(self):
        """Import gestures from a folder dataset (subfolder per gesture class)."""
        if not self.current_model_file:
            show_error_popup(
                self.window,
                "ยังไม่ได้เลือกโมเดล",
                "กรุณาสร้างหรือเปิดโมเดลก่อนนำเข้า Dataset"
            )
            return

        folder = filedialog.askdirectory(title="เลือก folder dataset")
        if not folder:
            return

        # Ask which hand type
        hand_type_popup = ctk.CTkToplevel(self.window)
        hand_type_popup.title("เลือกประเภทมือ")
        hand_type_popup.geometry("350x200")
        hand_type_popup.transient(self.window)
        hand_type_popup.grab_set()

        hand_type_popup.update_idletasks()
        x = (hand_type_popup.winfo_screenwidth() // 2) - 175
        y = (hand_type_popup.winfo_screenheight() // 2) - 100
        hand_type_popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            hand_type_popup,
            text="เลือกประเภทมือสำหรับ Dataset",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(20, 15))

        selected_hand = ctk.StringVar(value="right")

        btn_frame = ctk.CTkFrame(hand_type_popup, fg_color="transparent")
        btn_frame.pack(pady=5)

        def start_import(ht):
            selected_hand.set(ht)
            hand_type_popup.destroy()
            self._do_dataset_import(folder, ht)

        ctk.CTkButton(
            btn_frame,
            text="มือซ้าย",
            command=lambda: start_import("left"),
            width=100, height=40,
            fg_color="#4a9eff", hover_color="#3a8eef"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="มือขวา",
            command=lambda: start_import("right"),
            width=100, height=40,
            fg_color="#4a9eff", hover_color="#3a8eef"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            hand_type_popup,
            text="ยกเลิก",
            command=hand_type_popup.destroy,
            width=100, height=35,
            fg_color="#95a5a6", hover_color="#7f8c8d"
        ).pack(pady=10)

    def _do_dataset_import(self, folder, hand_type):
        """Perform the actual dataset import in a background thread."""
        # Show progress popup
        progress_popup = ctk.CTkToplevel(self.window)
        progress_popup.title("นำเข้า Dataset")
        progress_popup.geometry("450x250")
        progress_popup.transient(self.window)
        progress_popup.grab_set()

        progress_popup.update_idletasks()
        x = (progress_popup.winfo_screenwidth() // 2) - 225
        y = (progress_popup.winfo_screenheight() // 2) - 125
        progress_popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            progress_popup,
            text="กำลังนำเข้า Dataset...",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 10))

        progress_bar = ctk.CTkProgressBar(progress_popup, width=350)
        progress_bar.pack(pady=10, padx=40)
        progress_bar.set(0)

        status_label = ctk.CTkLabel(
            progress_popup,
            text="เริ่มต้น...",
            font=ctk.CTkFont(size=14)
        )
        status_label.pack(pady=5)

        result_label = ctk.CTkLabel(
            progress_popup,
            text="",
            font=ctk.CTkFont(size=13),
            justify="left"
        )
        result_label.pack(pady=10)

        close_btn = ctk.CTkButton(
            progress_popup,
            text="ปิด",
            command=progress_popup.destroy,
            state="disabled",
            width=120
        )
        close_btn.pack(pady=10)

        def do_import():
            # Create a temporary static Hands detector for image processing
            hands_detector = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5
            )

            def detect_fn(img):
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands_detector.process(rgb)
                if result.multi_hand_landmarks:
                    return result.multi_hand_landmarks[0]
                return None

            def extract_fn(landmarks):
                return HandMLP.extract_single_hand_features(landmarks)

            def progress_cb(current, total, class_name):
                try:
                    progress_bar.set(current / total if total > 0 else 0)
                    status_label.configure(
                        text=f"กำลังประมวลผล: {class_name} ({current}/{total})"
                    )
                except:
                    pass

            result = load_dataset_from_folder(folder, detect_fn, extract_fn, progress_cb)
            hands_detector.close()

            # Merge results into self.gestures
            total_gestures = 0
            total_samples = 0

            for gesture_name, features_list in result.items():
                if gesture_name in self.gestures[hand_type]:
                    # Append to existing gesture
                    self.gestures[hand_type][gesture_name].extend(features_list)
                else:
                    # New gesture
                    self.gestures[hand_type][gesture_name] = features_list
                total_gestures += 1
                total_samples += len(features_list)

            # Save model file
            if total_samples > 0:
                with open(self.current_model_file, 'w', encoding='utf-8') as f:
                    json.dump(self.gestures, f, ensure_ascii=False, indent=2)

            try:
                hand_name = "มือซ้าย" if hand_type == "left" else "มือขวา"
                progress_bar.set(1.0)
                status_label.configure(text="นำเข้าเสร็จสิ้น!")
                result_label.configure(
                    text=f"นำเข้า {total_gestures} ท่าทาง ({total_samples} ตัวอย่าง)\n"
                         f"ประเภท: {hand_name}"
                )
                close_btn.configure(state="normal")

                self.stats_label.configure(text=self.get_stats_text())
                self.update_gesture_list()
            except:
                pass

        threading.Thread(target=do_import, daemon=True).start()

    # =========================================================================
    # Video Capture & Processing
    # =========================================================================

    def capture_video(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, captured_frame = cap.read()
            if not ret:
                break
            captured_frame = cv2.flip(captured_frame, 1)
            image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            with self.lock:
                self.frame = captured_frame.copy()
                self.hand_landmarks = results.multi_hand_landmarks
                self.hand_handedness = results.multi_handedness

        cap.release()

    def process_frame(self):
        target_fps = 30
        frame_time = 1.0 / target_fps

        while self.running:
            start_time = time.time()

            local_frame = None
            local_hand_landmarks = None
            local_hand_handedness = None

            with self.lock:
                if self.frame is not None:
                    local_frame = self.frame.copy()
                    local_hand_landmarks = self.hand_landmarks
                    local_hand_handedness = self.hand_handedness

            if local_frame is not None:
                if self.current_mode != "compare":
                    left_gesture_text = "ไม่พบมือ"
                    right_gesture_text = "ไม่พบมือ"
                    both_gesture_text = "-"

                    if local_hand_landmarks and local_hand_handedness:
                        left_landmarks_ref = None
                        right_landmarks_ref = None

                        for idx, (landmarks, handedness) in enumerate(
                                zip(local_hand_landmarks, local_hand_handedness)):

                            if self.show_skeleton:
                                self.mp_drawing.draw_landmarks(
                                    local_frame,
                                    landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(
                                        color=(0, 255, 0), thickness=2, circle_radius=3),
                                    self.mp_drawing.DrawingSpec(
                                        color=(255, 0, 255), thickness=2)
                                )

                            hand_label = handedness.classification[0].label
                            hand_type = 'left' if hand_label == 'Left' else 'right'

                            # Use HandRecognition API for recognition
                            gesture, confidence = self.recognition.recognize_from_landmarks(
                                landmarks, hand_type, self.gestures
                            )

                            # Apply temporal smoothing at GUI level
                            if self.use_smoothing:
                                self.smoother.update(hand_type, gesture, confidence)
                                gesture, confidence = self.smoother.get_smoothed(hand_type)

                            if confidence > 0:
                                gesture_text = f"{gesture} ({confidence:.1f}%)"
                            else:
                                gesture_text = gesture

                            if hand_type == 'left':
                                left_gesture_text = gesture_text
                                left_landmarks_ref = landmarks
                            else:
                                right_gesture_text = gesture_text
                                right_landmarks_ref = landmarks

                        # Both-hands recognition
                        if left_landmarks_ref is not None and right_landmarks_ref is not None:
                            both_gestures_data = self.gestures.get('both', {})
                            if both_gestures_data:
                                both_g, both_c = self.recognition.recognize_both_from_landmarks(
                                    left_landmarks_ref, right_landmarks_ref, self.gestures
                                )
                                if self.use_smoothing:
                                    self.smoother.update('both', both_g, both_c)
                                    both_g, both_c = self.smoother.get_smoothed('both')
                                if both_c > 0:
                                    both_gesture_text = f"{both_g} ({both_c:.1f}%)"
                                else:
                                    both_gesture_text = both_g

                        if self.show_text_overlay:
                            mode_text = f"[{self.recognition_mode.upper()}]"
                            local_frame = draw_thai_text(
                                local_frame,
                                f"{mode_text} มือซ้าย: {left_gesture_text}",
                                (15, 15),
                                self.thai_font_video,
                                (0, 255, 0)
                            )
                            local_frame = draw_thai_text(
                                local_frame,
                                f"{mode_text} มือขวา: {right_gesture_text}",
                                (15, 60),
                                self.thai_font_video,
                                (0, 255, 0)
                            )
                            if both_gesture_text != "-":
                                local_frame = draw_thai_text(
                                    local_frame,
                                    f"{mode_text} 2 มือ: {both_gesture_text}",
                                    (15, 105),
                                    self.thai_font_video,
                                    (0, 165, 255)
                                )
                    else:
                        if self.use_smoothing:
                            self.smoother.reset()
                        if self.show_text_overlay:
                            local_frame = draw_thai_text(
                                local_frame,
                                "มือซ้าย: ไม่พบมือ",
                                (15, 15),
                                self.thai_font_video,
                                (0, 0, 255)
                            )
                            local_frame = draw_thai_text(
                                local_frame,
                                "มือขวา: ไม่พบมือ",
                                (15, 60),
                                self.thai_font_video,
                                (0, 0, 255)
                            )

                    # Draw countdown overlay on video
                    if self.countdown_active and self.countdown_value > 0:
                        h, w = local_frame.shape[:2]
                        overlay = local_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.4, local_frame, 0.6, 0, local_frame)
                        countdown_text = str(self.countdown_value)
                        font_scale = 8
                        thickness = 15
                        text_size = cv2.getTextSize(
                            countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )[0]
                        text_x = (w - text_size[0]) // 2
                        text_y = (h + text_size[1]) // 2
                        cv2.putText(
                            local_frame, countdown_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness
                        )

                    # Update GUI labels
                    try:
                        self.left_gesture_label.configure(text=left_gesture_text)
                        self.right_gesture_label.configure(text=right_gesture_text)
                        self.both_gesture_label.configure(text=both_gesture_text)
                    except:
                        pass

                else:  # compare mode
                    if local_hand_landmarks:
                        for landmarks in local_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                local_frame,
                                landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(
                                    color=(0, 255, 0), thickness=2, circle_radius=3),
                                self.mp_drawing.DrawingSpec(
                                    color=(255, 0, 255), thickness=2)
                            )

                    model1_left = "ไม่พบมือ"
                    model1_right = "ไม่พบมือ"
                    model2_left = "ไม่พบมือ"
                    model2_right = "ไม่พบมือ"

                    if local_hand_landmarks and local_hand_handedness:
                        for landmarks, handedness in zip(
                                local_hand_landmarks, local_hand_handedness):
                            hand_label = handedness.classification[0].label
                            hand_type = 'left' if hand_label == 'Left' else 'right'

                            # Use HandRecognition API for compare mode
                            gesture1, conf1 = self.recognition.recognize_from_landmarks(
                                landmarks, hand_type, self.model1_gestures
                            )
                            text1 = f"{gesture1} ({conf1:.1f}%)" if conf1 > 0 else gesture1

                            gesture2, conf2 = self.recognition.recognize_from_landmarks(
                                landmarks, hand_type, self.model2_gestures
                            )
                            text2 = f"{gesture2} ({conf2:.1f}%)" if conf2 > 0 else gesture2

                            if hand_type == 'left':
                                model1_left = text1
                                model2_left = text2
                            else:
                                model1_right = text1
                                model2_right = text2

                    # Update GUI labels
                    try:
                        self.model1_left_label.configure(text=model1_left)
                        self.model1_right_label.configure(text=model1_right)
                        self.model2_left_label.configure(text=model2_left)
                        self.model2_right_label.configure(text=model2_right)
                    except:
                        pass

                # Prepare image for display
                self.update_canvas_size()

                h, w = local_frame.shape[:2]
                canvas_w = self.canvas_width
                canvas_h = self.canvas_height

                scale = min(canvas_w / w, canvas_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                resized_frame = cv2.resize(
                    local_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

                display_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                y_offset = (canvas_h - new_h) // 2
                x_offset = (canvas_w - new_w) // 2
                display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

                image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image=image)

                if self.canvas_image_id is None:
                    self.canvas_image_id = self.canvas.create_image(
                        canvas_w // 2, canvas_h // 2, image=photo, anchor="center"
                    )
                else:
                    self.canvas.itemconfig(self.canvas_image_id, image=photo)
                    self.canvas.coords(self.canvas_image_id, canvas_w // 2, canvas_h // 2)

                self.canvas.image = photo

            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # =========================================================================
    # GUI update & cleanup
    # =========================================================================

    def update_gui(self):
        if self.running:
            self.window.after(33, self.update_gui)

    def on_closing(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)
        try:
            self.window.quit()
            self.window.destroy()
        except:
            pass


if __name__ == "__main__":
    app = GestureModelManager(ctk.CTk())
    app.window.mainloop()

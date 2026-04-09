"""
Shared Utilities
=================
Common helpers shared across hand, face, body modules
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Callable, Dict, List, Tuple, Optional


# ---- Font loading ----
#
# The Thai-capable TrueType font is NOT shipped with this repository.
# By default, the library searches these locations (in order) for a font file:
#   1. The $GESTURE_FONT_PATH environment variable (absolute path).
#   2. ./assets/<filename>       (drop your own font here)
#   3. ./no_upto_git/<filename>  (local-only scratch folder, gitignored)
# If nothing is found, PIL's built-in bitmap font is used as a last resort
# — Thai glyphs will not render correctly in that case.
#
# To point at a specific font file from your own code:
#
#     from hand.hand_recognition import HandRecognition
#     rec = HandRecognition(font_path="C:/path/to/your/thai-font.ttf")
#
# or set the environment variable once:
#
#     set GESTURE_FONT_PATH=C:\fonts\NotoSansThai-Regular.ttf    # Windows
#     export GESTURE_FONT_PATH=/usr/share/fonts/NotoSansThai.ttf  # Linux/macOS

DEFAULT_FONT_FILENAME = "AmericanoDemo.ttf"


def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to PIL's default if unavailable."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def get_asset_path(filename: str = DEFAULT_FONT_FILENAME) -> Optional[str]:
    """Resolve a font/asset path by checking env var, assets/, then no_upto_git/.

    Returns the first path that exists on disk, or None if none were found.
    """
    env_path = os.environ.get("GESTURE_FONT_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "assets", filename),
        os.path.join(project_root, "no_upto_git", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ---- Drawing helpers ----

def draw_thai_text(image: np.ndarray, text: str, position: Tuple[int, int],
                   font: ImageFont.FreeTypeFont,
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """วาดข้อความด้วย PIL (รองรับ Thai font), input/output เป็น BGR"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_countdown_overlay(frame: np.ndarray, value: int,
                           font: ImageFont.FreeTypeFont) -> np.ndarray:
    """วาด countdown overlay (3, 2, 1) ตรงกลางภาพ"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    text = str(value)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((w - tw) // 2, (h - th) // 2)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def display_frame_on_canvas(frame: np.ndarray, canvas, canvas_w: int, canvas_h: int):
    """Resize frame with letterbox and display on tkinter canvas"""
    from PIL import ImageTk

    h, w = frame.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create letterbox canvas
    canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    canvas_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    img_rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    canvas.delete("all")
    canvas.create_image(canvas_w // 2, canvas_h // 2, image=img_tk, anchor="center")
    canvas._img_ref = img_tk  # prevent GC


# ---- Popups ----

def show_error_popup(window, title: str, message: str):
    """แสดง error popup"""
    import customtkinter as ctk

    popup = ctk.CTkToplevel(window)
    popup.title(title)
    popup.geometry("400x200")
    popup.transient(window)
    popup.grab_set()

    ctk.CTkLabel(popup, text=title, font=ctk.CTkFont(size=18, weight="bold"),
                 text_color="#e74c3c").pack(pady=(20, 10))
    ctk.CTkLabel(popup, text=message, font=ctk.CTkFont(size=14),
                 wraplength=350).pack(pady=10)
    ctk.CTkButton(popup, text="ตกลง", command=popup.destroy,
                  width=100).pack(pady=10)


def show_success_popup(window, message: str, auto_close_ms: int = 1500):
    """แสดง success popup แล้วปิดอัตโนมัติ"""
    import customtkinter as ctk

    popup = ctk.CTkToplevel(window)
    popup.title("สำเร็จ")
    popup.geometry("300x120")
    popup.transient(window)

    ctk.CTkLabel(popup, text=message, font=ctk.CTkFont(size=16),
                 text_color="#2ecc71").pack(expand=True, pady=20)

    popup.after(auto_close_ms, popup.destroy)


# ---- Dataset loader ----

def load_dataset_from_folder(folder_path: str,
                             detect_fn: Callable[[np.ndarray], object],
                             extract_fn: Callable[[object], Optional[np.ndarray]],
                             progress_cb: Callable[[int, int, str], None] = None
                             ) -> Dict[str, List[list]]:
    """
    โหลด dataset จาก folder structure:
    folder_path/
    ├── class_name_1/
    │   ├── img1.jpg
    │   └── img2.png
    └── class_name_2/
        └── img1.jpg

    Args:
        detect_fn: function(bgr_image) → landmarks object หรือ None
        extract_fn: function(landmarks) → feature_vector (np.ndarray) หรือ None
        progress_cb: callback(current_count, total_count, class_name)

    Returns:
        dict: {class_name: [[feature_vector], ...]}
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    result = {}

    if not os.path.isdir(folder_path):
        return result

    # Count total images
    total = 0
    class_dirs = []
    for name in sorted(os.listdir(folder_path)):
        sub = os.path.join(folder_path, name)
        if os.path.isdir(sub):
            imgs = [f for f in os.listdir(sub) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            if imgs:
                class_dirs.append((name, sub, imgs))
                total += len(imgs)

    current = 0
    for class_name, class_dir, image_files in class_dirs:
        features_list = []

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                current += 1
                continue

            landmarks = detect_fn(img)
            if landmarks is not None:
                features = extract_fn(landmarks)
                if features is not None:
                    features_list.append(features.tolist())

            current += 1
            if progress_cb:
                progress_cb(current, total, class_name)

        if features_list:
            result[class_name] = features_list

    return result

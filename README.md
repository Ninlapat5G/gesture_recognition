# Gesture Recognition

A lightweight, dependency-light toolkit for building custom **hand gesture**,
**body pose**, and **face expression** classifiers on top of Google MediaPipe.
Everything — landmark extraction, feature engineering, the MLP classifier
itself — runs on plain NumPy, so there is no PyTorch / TensorFlow install
to wrestle with.

The project started as a weekend experiment and slowly grew into three
parallel pipelines that all share the same core: extract landmarks, turn them
into a rotation- and scale-invariant feature vector, then either match
against a stored template (MAE mode) or feed the vector through a tiny MLP
(MLP mode). The tkinter GUIs on top are there to make data capture and
training painless — point a webcam, hit record, type a label.

## Why another gesture lib?

There are a few reasons I kept building on this instead of reaching for a
bigger framework:

- **No heavyweight ML dependency.** The classifier is a 200-line NumPy MLP
  with Adam, dropout, He init, stratified split, and early stopping. It
  trains in seconds on a laptop CPU and serializes to a single JSON file.
- **Two modes, one API.** MAE matching works the instant you've captured two
  or three samples of a gesture — handy for prototyping. When you have more
  data and want better generalization, flip `mode="mlp"` and train.
- **Scale / rotation invariant features.** Distances are normalized by a
  skeleton-relative size (hand width, shoulder width, inter-ocular distance),
  and joint angles are included so the model doesn't get confused by camera
  distance or small tilts.
- **Temporal smoothing out of the box.** A weighted voting window reduces
  the label flicker you get from raw per-frame predictions.

## Features

### Hand gestures (`hand/`)
- Single-hand classifier with **225-dim features**: 210 pairwise distances
  (21 landmarks) + 15 joint angles across the five fingers.
- Two-hands classifier with **861-dim features**: left + right + 441
  cross-hand distances, normalized by average hand size.
- Position tracking: read a configurable landmark (wrist, index tip, middle
  MCP…) to get a scaled (x, y) offset since the gesture first appeared.
- Backwards-compatible: can still load older 210-dim "distances only" models.

### Body pose (`body/`)
- 33-landmark MediaPipe Pose pipeline.
- **548-dim features**: 528 pairwise distances normalized by shoulder width
  + 20 anatomically meaningful joint angles (shoulders, elbows, hips, knees,
  ankles, torso, arm-to-hip).
- Landmark quality gate — drops frames where key joints are low-visibility
  or the subject is too far from the camera.

### Face expressions (`face/`)
- 50 key Face Mesh landmarks selected from eyes, brows, nose, mouth, jaw.
- **1265-dim features**: 1225 pairwise distances + 25 facial angles + 15
  expression-specific ratios (eye aspect, mouth aspect, brow-to-eye, etc.).
- Optional 2D / 3D mode — drop the z-axis if your capture is noisy.

### Shared goodies (`core/`)
- `BaseMLP`: reusable NumPy classifier. Adam, inverted dropout, He init,
  stratified train/val split (80/20), adaptive Gaussian augmentation
  (5× – 30× depending on sample count), early stopping on val loss.
- `TemporalSmoother`: weighted-vote deque, O(1) per frame, string-keyed so
  left / right / both hands each get their own history.
- Thai-text drawing helper (PIL), dataset loader that walks
  `folder/class_name/image.jpg`, popup helpers, frame-to-canvas display with
  letterboxing.

### Full GUI apps
- `GUI_hand.py`, `GUI_body.py`, `GUI_face.py` launch customtkinter apps for
  capturing samples, training MLPs with a live progress chart, comparing
  models on a held-out set, and running real-time recognition from a webcam.

## Installation

```bash
git clone https://github.com/<your-username>/gesture_recognition.git
cd gesture_recognition

# Option A: editable install (recommended for development)
pip install -e ".[gui]"

# Option B: plain install
pip install ".[gui]"

# Option C: just the library, no GUI
pip install .
```

Python 3.10+ is recommended (MediaPipe doesn't play well with older or very
new Python versions — see the [MediaPipe release notes][mp-rel] if pip
complains).

[mp-rel]: https://github.com/google-ai-edge/mediapipe/releases

### Thai font (optional)

The project can draw Thai labels on the output frames, but the TTF file
itself is **not** shipped with the repo (it lived under a restrictive
license). The code will fall back to PIL's built-in bitmap font if no TTF
is found — everything still works, just with plain English glyphs.

If you want Thai rendering, drop any Thai-capable TrueType font into
`assets/` or point the library at a font of your choosing:

```python
from gesture_recognition.hand import HandRecognition

# Option 1: pass it explicitly
rec = HandRecognition(font_path="C:/Windows/Fonts/Tahoma.ttf")

# Option 2: set an environment variable once
# set GESTURE_FONT_PATH=C:\Windows\Fonts\Tahoma.ttf   (Windows)
# export GESTURE_FONT_PATH=/usr/share/fonts/truetype/tlwg/Garuda.ttf   (Linux)
```

## Quick start

```python
import cv2
from gesture_recognition.hand import HandRecognition

rec = HandRecognition(mode="mae", smoothing_window=5)
model = rec.load_model("my_gestures.json")  # captured via GUI_hand.py

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    status, gestures, confidences, annotated = rec.predict_frame(
        model, frame, show_gesture=1
    )
    cv2.imshow("Hand", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

More runnable snippets (webcam + still image for each of the three modules)
live in the [`example/`](example/) folder.

## Training your own model

1. **Capture data.** Launch `python GUI_hand.py` (or `GUI_body.py` /
   `GUI_face.py`), create a new model, record 10–30 frames per gesture.
   After `pip install -e ".[gui]"` you can also use the console commands
   `gesture-hand`, `gesture-body`, `gesture-face`.
2. **Switch to MLP mode** in the GUI and hit *Train*. The training curve
   and final validation accuracy are displayed live.
3. **Save.** The GUI writes two files:
   - `my_gestures.json` — the raw captured feature vectors (human readable)
   - `my_gestures_left.mlp.json`, `..._right.mlp.json`, `..._both.mlp.json`
     — the trained MLP weights (whichever hand types had ≥2 classes)
4. **Load at runtime** with `rec.load_model(...)` + `rec.load_mlp(...)`.

The same flow works for body poses and face expressions — the only thing
that changes is which module you import.

## Project layout

```
gesture_recognition/
├── src/
│   └── gesture_recognition/   # installable package
│       ├── core/              # BaseMLP, TemporalSmoother, utils
│       ├── hand/              # Hand pipeline (MLP + API + GUI)
│       ├── body/              # Body pose pipeline
│       └── face/              # Face expression pipeline
├── example/                   # Minimal runnable usage examples
├── GUI_hand.py                # GUI entry point (dev shortcut)
├── GUI_body.py
├── GUI_face.py
├── pyproject.toml             # pip install config
├── requirements.txt
└── README.md
```

`no_upto_git/` is a scratch folder for anything you want to keep locally but
not push to GitHub — the Thai font, personal training data, trained weights,
etc. It's listed in `.gitignore`.

## Architecture notes

**Feature design.** All three classifiers follow the same recipe: compute
every pairwise distance between the relevant landmarks, divide by a
skeleton-relative length so zoom doesn't matter, then append a handful of
angles (and for faces, a handful of ratios) so the model has some
rotation-invariant structure to latch onto.

**MLP.** Two hidden layers by default — `[128, 64]` for small inputs, scaled
to `[256, 128]` for the larger (both-hands / body / face) feature vectors.
Trained with Adam (lr 0.001), dropout 0.3, batch size 32, up to 300 epochs
with early stopping (patience 20) on validation loss. Training data is
augmented 5–30× with Gaussian noise proportional to the per-feature std;
the multiplier scales with dataset size so tiny datasets still get enough
effective samples.

**MAE mode.** For the "I just captured three examples, does it work?" case,
each frame is scored against stored templates via mean absolute error
between normalized feature vectors. Body and face both weight the angle /
ratio terms more heavily than raw distances, which keeps noisy distance
components from drowning out the discriminative signal.

## Contributing

Pull requests, bug reports and feature ideas are all welcome. If you have a
domain the current feature extractors don't cover (e.g. a foot pose, or a
single-finger tracking mode), the cleanest extension path is:

1. Subclass `gesture_recognition.core.BaseMLP` in a new module.
2. Implement `calculate_distances`, `calculate_angles` and `extract_features`.
3. Write a thin high-level API wrapping it, mirroring `hand_recognition.py`.

## License

No explicit license — this is a small-group project for internal use. If
you want to reuse it beyond that, just open an issue and ask.
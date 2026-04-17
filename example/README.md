# Examples

Minimal, runnable snippets showing how to consume each module as a library
— no GUI, no frills. Each script assumes you've already captured a model
with the matching GUI.

| Script               | What it does                                        |
|----------------------|-----------------------------------------------------|
| `hand_webcam.py`     | Real-time hand gesture recognition from a webcam    |
| `hand_image.py`      | Run hand recognition on a single image file         |
| `body_webcam.py`     | Real-time body pose recognition from a webcam       |
| `face_webcam.py`     | Real-time face expression recognition from a webcam |

## One-time setup

From the repo root:

```bash
pip install -e ".[gui]"
```

That gives you the `gesture_recognition` package plus the GUI extras
(`customtkinter`). Skip `[gui]` if you only need the library.

## Capture a model first

Every example reads a JSON model produced by the capture GUI. Launch the
matching GUI, record a few samples per class, and save:

```bash
python run_gui.py hand      # → my_gestures.json
python run_gui.py body      # → my_poses.json
python run_gui.py face      # → my_expressions.json
```

Adjust the `MODEL_PATH` constant at the top of the example script if you
save under a different name.

## Run

```bash
python example/hand_webcam.py
python example/hand_image.py path/to/photo.jpg
python example/body_webcam.py
python example/face_webcam.py
```

Press **ESC** to quit the webcam examples. If a model file is missing,
the script prints a friendly error pointing you back to the capture step.

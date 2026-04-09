# Examples

Minimal, runnable snippets showing how to consume each module as a library
— no GUI, no frills. Each script assumes you already have a trained model
file (captured and saved from the corresponding `run_*.py` GUI app).

| Script                       | What it does                                       |
|------------------------------|----------------------------------------------------|
| `hand_webcam.py`             | Real-time hand gesture recognition from a webcam   |
| `hand_image.py`              | Run hand recognition on a single image file        |
| `body_webcam.py`             | Real-time body pose recognition from a webcam      |
| `face_webcam.py`             | Real-time face expression recognition from a webcam|

Run them from the **project root** so the `hand / body / face / core`
imports resolve:

```bash
cd gesture_recognition
python example/hand_webcam.py
```

If you haven't captured a model yet, launch the matching GUI first:

```bash
python run_hand.py      # then capture gestures and save as my_gestures.json
```

and edit the `MODEL_PATH` constant at the top of the example script to
point at the file you just saved.

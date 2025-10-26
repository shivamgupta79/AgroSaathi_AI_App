"""
app.py - Flask + webcam + YOLO (robust error handling)

Notes:
- Works with ultralytics YOLO (v8+). If you use a different API, adapt the model.predict call.
- If webcam isn't available, it will try fallback to video file "static/fallback.mp4" if present.
- Logs errors to console and returns a simple error image stream so browser doesn't hang.
"""

import io
import time
import sys
import traceback
from flask import Flask, render_template, Response
import cv2
import numpy as np

# Try to import ultralytics; handle import error gracefully
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("Warning: ultralytics not installed or failed to import:", e, file=sys.stderr)

app = Flask(__name__)

# CONFIG
MODEL_PATH = "models/best.pt"           # adjust if different
WEBCAM_INDEX = 0                        # try 0,1,2 if needed
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
USE_GPU = False                          # set True if you want to force CUDA (and have it installed)


def load_model(path):
    """Load YOLO model, with clear errors."""
    if YOLO is None:
        raise RuntimeError("ultralytics package not found. Install it (pip install ultralytics).")
    try:
        model = YOLO(path)
        # Optionally warm-up a dummy prediction to catch model issues early
        # but avoid heavy compute on init
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{path}': {e}")


# Try to load model (but do not crash the app — we will show helpful message in UI)
model = None
model_load_error = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model_load_error = str(e)
    print("Model load error:", model_load_error, file=sys.stderr)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    # set resolution (may or may not be supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

# Initialize capture
cap = open_camera(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Warning: webcam at index {WEBCAM_INDEX} not opened.", file=sys.stderr)


def make_error_frame(text, width=640, height=480):
    """Return a JPEG bytes of an error image with text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # put the message
    lines = text.split('\n')
    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ret, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ret else b''


def gen_frames():
    """Generator that yields JPEG frames for Flask Response."""
    global cap, model, model_load_error

    # If camera not opened, try fallback video
    if not cap.isOpened():
        fallback_path = "static/fallback.mp4"
        try:
            cap = cv2.VideoCapture(fallback_path)
            print("Using fallback video:", fallback_path)
        except Exception:
            pass

    # If still not open, stream error image repeatedly
    if not cap.isOpened():
        err = "ERROR: Webcam not available.\nCheck camera index or device permissions."
        frame = make_error_frame(err)
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.5)

    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("Frame read failed (end of stream or camera disconnected). Retrying in 0.5s...", file=sys.stderr)
                time.sleep(0.5)
                continue

            # If model failed to load earlier, show frame but overlay message
            if model is None:
                overlaid = frame.copy()
                cv2.putText(overlaid, "Model not loaded. Check server logs.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', overlaid)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            # Ultralytics YOLO accepts BGR numpy arrays; but to be safe, call model.predict
            # Use a lightweight call (no large resizing) and catch exceptions per-frame
            try:
                # For larger models you may want to reduce imgsz to speed up
                results = model.predict(source=frame, conf=0.25, verbose=False, device='cuda' if USE_GPU else 'cpu')
                # results is a Results object or list — we handle both cases
                r0 = results[0] if isinstance(results, (list, tuple)) else results
                # draw boxes
                try:
                    annotated = r0.plot()
                    # r0.plot() returns an ndarray BGR
                except Exception:
                    # fallback: just use original frame if plot fails
                    annotated = frame
            except Exception as yerr:
                # If YOLO prediction crashes, show original frame with error text
                tb = traceback.format_exc(limit=1)
                print("YOLO prediction error (continuing):", yerr, file=sys.stderr)
                print(tb, file=sys.stderr)
                annotated = frame.copy()
                cv2.putText(annotated, "YOLO error - see server logs", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # encode and yield
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                # fallback error image
                frame = make_error_frame("Error encoding frame")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            # Client disconnected
            print("Client disconnected; stopping frame generator.")
            break
        except Exception:
            print("Unhandled error in frame generation:", file=sys.stderr)
            traceback.print_exc()
            time.sleep(0.5)


@app.route('/')
def index():
    return render_template('index.html', model_error=model_load_error)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def cleanup():
    global cap
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass


if __name__ == '__main__':
    try:
        print("Starting AgroSaathi AI app...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cleanup()

"""Display face (eyes) on a 128x32 RGB LED matrix using MediaPipe FaceMesh.

This is a simplified adaptation of preview_face.py for Raspberry Pi RGB matrices.

Requirements (install on Raspberry Pi):
  sudo apt install python3-picamera2 (if using Pi Camera)
  pip install pillow opencv-python mediapipe
  pip install rpi-rgb-led-matrix (or build from source per project docs)

Environment variables (optional):
  BRIGHTNESS        1-100 (default 60)
  CAM_W, CAM_H      Capture request (default 640x480)
  CAM_FPS           Target camera FPS (30)
  PROC_SCALE        Downscale before FaceMesh (1.0 = full res)
  SMOOTH_ALPHA      0..1 eye ratio smoothing (0.3)
  REFINE            0/1 enable iris refine (1)
  SHOW_FPS          0/1 print FPS to console
  FOURCC_MJPG       0/1 try MJPG for USB cam (1)
"""

from __future__ import annotations

import os, time, math, threading, queue
from typing import Any, TYPE_CHECKING

from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

try:  # Matrix library (only available on Pi)
	from rgbmatrix import RGBMatrix, RGBMatrixOptions  # type: ignore
except Exception as e:  # pragma: no cover - informative fallback
	RGBMatrix = RGBMatrixOptions = None  # type: ignore
	_MATRIX_IMPORT_ERROR = e

if TYPE_CHECKING:  # hint only
	Picamera2 = Any  # type: ignore

# ---------- Configuration ----------
BRIGHTNESS = int(os.environ.get("BRIGHTNESS", "60"))
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))
PROC_SCALE = float(os.environ.get("PROC_SCALE", "1.0"))
PROC_SCALE = max(0.2, min(1.0, PROC_SCALE))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", "0.3"))
REFINE = os.environ.get("REFINE", "1") == "1"
SHOW_FPS = os.environ.get("SHOW_FPS", "0") == "1"
FOURCC_MJPG = os.environ.get("FOURCC_MJPG", "1") == "1"

# ---------- Assets (eye frames) ----------
EYE_FILES = ["protogen.png", "protogen1.png", "protogen2.png", "protogen3.png"]
EYE_FRAMES: list[Image.Image] = []
for f in EYE_FILES:
	if not os.path.exists(f):
		raise FileNotFoundError(f"Missing eye frame image: {f}")
	EYE_FRAMES.append(Image.open(f).convert("RGBA"))

# ---------- FaceMesh setup ----------
def build_facemesh(refine: bool):
	return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=refine)

mp_face = build_facemesh(REFINE)

# Landmarks indices (swapped to match display orientation like preview)
L = [362, 385, 387, 263, 373, 380]  # left eye (display-left)
R = [33, 160, 158, 133, 153, 144]   # right eye (display-right)

def eye_ratio(landmarks, idxs):  # type: ignore
	pts = [landmarks[i] for i in idxs]
	d = lambda a, b: math.dist((a.x, a.y), (b.x, b.y))
	v = (d(pts[1], pts[5]) + d(pts[2], pts[4])) / 2
	h = d(pts[0], pts[3])
	return v / h if h else 0.0

def eye_index(r: float) -> int:
	# Same thresholds as preview
	return 0 if r > 0.30 else 1 if r > 0.24 else 2 if r > 0.19 else 3

# ---------- Camera (Picamera2 preferred) ----------
try:
	from picamera2 import Picamera2  # type: ignore
	_HAVE_PICAMERA2 = True
except Exception:
	Picamera2 = None  # type: ignore
	_HAVE_PICAMERA2 = False

def open_camera():
	if _HAVE_PICAMERA2:
		try:
			cam = Picamera2()
			cfg = cam.create_video_configuration(main={"size": (CAM_W, CAM_H)})
			cam.configure(cfg)
			cam.start()
			return cam, None
		except Exception:
			pass
	cap = cv2.VideoCapture(0)
	if cap.isOpened():
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
		cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
		if FOURCC_MJPG:
			try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
			except Exception: pass
		try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		except Exception: pass
		return None, cap
	raise RuntimeError("Unable to open camera")

# ---------- Matrix init ----------
def init_matrix():
	if RGBMatrix is None or RGBMatrixOptions is None:  # pragma: no cover
		raise ImportError(f"rgbmatrix library not available: {_MATRIX_IMPORT_ERROR}")
	opts = RGBMatrixOptions()
	opts.rows = 32
	opts.cols = 128
	opts.hardware_mapping = "adafruit-hat"
	if hasattr(opts, "disable_hardware_pulsing"):
		opts.disable_hardware_pulsing = True
	elif hasattr(opts, "no_hardware_pulse"):
		opts.no_hardware_pulse = True
	if hasattr(opts, "brightness"):
		opts.brightness = BRIGHTNESS
	return RGBMatrix(options=opts)

# ---------- Main loop ----------
def run():
	matrix = init_matrix()
	picam, cap = open_camera()

	# Optional thread for OpenCV capture to reduce latency
	frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
	stop_flag = False

	def capture_loop():
		while not stop_flag:
			ok, f = cap.read()  # type: ignore
			if not ok:
				time.sleep(0.01)
				continue
			if frame_q.full():
				try: frame_q.get_nowait()
				except Exception: pass
			try: frame_q.put_nowait(f)
			except Exception:
				pass

	thread = None
	if cap is not None:
		thread = threading.Thread(target=capture_loop, daemon=True)
		thread.start()

	last_li = last_ri = -1
	plr = prr = None  # smoothed ratios
	fps_t0 = time.time(); frames = 0

	print("Started face display. Press Ctrl+C to exit.")
	try:
		while True:
			# Acquire frame
			if picam is not None:
				try:
					frame = picam.capture_array()
					frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
				except Exception:
					continue
			else:
				try:
					frame = frame_q.get(timeout=1.0) if thread else cap.read()[1]  # type: ignore
				except Exception:
					continue

			proc = frame
			if PROC_SCALE < 1.0:
				new_w = max(64, int(proc.shape[1] * PROC_SCALE))
				new_h = max(48, int(proc.shape[0] * PROC_SCALE))
				proc = cv2.resize(proc, (new_w, new_h), interpolation=cv2.INTER_AREA)

			res = mp_face.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
			li, ri = last_li, last_ri
			if res.multi_face_landmarks:
				lms = res.multi_face_landmarks[0].landmark
				lr = eye_ratio(lms, L)
				rr = eye_ratio(lms, R)
				if plr is not None:
					lr = plr * (1 - SMOOTH_ALPHA) + lr * SMOOTH_ALPHA
					rr = prr * (1 - SMOOTH_ALPHA) + rr * SMOOTH_ALPHA
				plr, prr = lr, rr
				li, ri = eye_index(lr), eye_index(rr)

			if (li, ri) != (last_li, last_ri) and li >= 0 and ri >= 0:
				canvas = Image.new("RGBA", (128, 32))
				left = EYE_FRAMES[li]
				right = EYE_FRAMES[ri].transpose(Image.FLIP_LEFT_RIGHT)
				canvas.paste(left, (0, 0), left)
				canvas.paste(right, (64, 0), right)
				# Push to matrix (PIL image is accepted by SetImage)
				matrix.SetImage(canvas.convert("RGB"), 0, 0)
				last_li, last_ri = li, ri

			frames += 1
			if SHOW_FPS and frames % 30 == 0:
				now = time.time()
				fps = 30.0 / (now - fps_t0) if now > fps_t0 else 0
				fps_t0 = now
				print(f"FPS ~ {fps:5.1f}  eye L/R idx=({last_li},{last_ri})  ratios=" +
					  (f" {plr:.2f}/{prr:.2f}" if plr is not None else " n/a"))

	except KeyboardInterrupt:
		print("Exiting...")
	finally:
		try: mp_face.close()
		except Exception: pass
		if picam is not None:
			try: picam.stop()
			except Exception: pass
		if cap is not None:
			try: cap.release()
			except Exception: pass


def main():  # entry point
	run()


if __name__ == "__main__":
	main()

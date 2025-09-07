"""Independent eye blink animation on 128x32 matrix.
Deps: pip install pillow opencv-python mediapipe rgbmatrix
On Raspberry Pi (Arducam/RPi Cam v2), also install: sudo apt install python3-picamera2
"""
from PIL import Image
import os
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import cv2, mediapipe as mp, math, time, os
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
	Picamera2 = Any  # type: ignore

# Prefer Pi Camera (Picamera2) on Raspberry Pi; fallback to USB/DirectShow/Video4Linux
try:
	from picamera2 import Picamera2  # type: ignore
	_HAVE_PICAMERA2 = True
except Exception:
	Picamera2 = None  # type: ignore
	_HAVE_PICAMERA2 = False

files=["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
eye_frames=[Image.open(f).convert("RGBA") for f in files]  # single-eye (64x32) art

# ---------------- Performance Tunables (override via env) ----------------
CAM_WIDTH  = int(os.environ.get("CAM_W", 320))   # lower = faster
CAM_HEIGHT = int(os.environ.get("CAM_H", 240))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL", 0.15))  # seconds between face mesh runs (~6-7 FPS)
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.3))  # lower = smoother
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW", "0") == "1"  # set NO_PREVIEW=1 to hide OpenCV window
REFINE_LANDMARKS = os.environ.get("REFINE", "0") == "1"  # off by default for speed
PROCESS_SCALE = float(os.environ.get("PROC_SCALE", 1.0))  # extra scale factor (e.g. 0.75) on resized frame

# -------------------------------------------------------------------------

opts = RGBMatrixOptions()
opts.rows = 32
opts.cols = 128            # if you actually have two 64x32 panels, use cols=64 and opts.chain_length = 2
opts.hardware_mapping = "adafruit-hat"
opts.no_hardware_pulse = True  # reduce CPU jitter
matrix = RGBMatrix(options=opts)

# Damaged columns mitigation (e.g., broken capacitor causing green tint).
# Configure via env COL_FIX: 'off' | 'black' | 'drop_green' | 'reduce_green'
BAD_COLUMNS = tuple(range(34, 39))  # inclusive: 34..38
COL_FIX_MODE = os.environ.get('COL_FIX', 'drop_green')

def apply_column_filter(img: Image.Image) -> Image.Image:
	if COL_FIX_MODE == 'off' or not BAD_COLUMNS:
		return img.convert("RGB")
	im = img.convert("RGB")
	px = im.load()
	w, h = im.size
	for x in BAD_COLUMNS:
		if 0 <= x < w:
			for y in range(h):
				r, g, b = px[x, y]
				if COL_FIX_MODE == 'black':
					px[x, y] = (0, 0, 0)
				elif COL_FIX_MODE == 'reduce_green':
					px[x, y] = (r, max(0, g // 4), b)
				else:  # 'drop_green'
					px[x, y] = (r, 0, b)
	return im

mp_face=mp.solutions.face_mesh.FaceMesh(
	max_num_faces=1,
	refine_landmarks=REFINE_LANDMARKS,
	static_image_mode=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)
"""Landmark order: [outer, top1, top2, inner, bottom2, bottom1] so (1,5) & (2,4) are vertical pairs.
Updated indices to fix right eye reading.
"""
# Swap to match display orientation (camera vs subject)
L=[362,385,387,263,373,380]          # left eye (display-left)
R=[33,160,158,133,153,144]           # right eye (display-right)

def eye_ratio(lms,ids):
	pts=[lms[i] for i in ids]; d=lambda a,b:math.dist((a.x,a.y),(b.x,b.y))
	v=(d(pts[1],pts[5])+d(pts[2],pts[4]))/2; h=d(pts[0],pts[3]); return v/h if h else 0

def idx(r): return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

# Initialize camera
picam = None  # type: ignore[assignment]
cap = None  # type: ignore[assignment]
if _HAVE_PICAMERA2:
	picam = Picamera2()
	# Lower resolution for Pi Zero performance
	RES = (CAM_WIDTH, CAM_HEIGHT)
	cfg = picam.create_video_configuration(main={"size": RES})
	picam.configure(cfg)
	picam.start()
else:
	cap = cv2.VideoCapture(0)
	if cap.isOpened():
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
		cap.set(cv2.CAP_PROP_FPS, 15)

# Single-threaded OpenCV (reduces overhead on single-core Pi Zero)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

last_process_time = 0.0
last_landmarks = None
last=(-1,-1)
while True:
	# Grab a frame from the active camera
	if picam is not None:
		try:
			frame = picam.capture_array()  # RGB
			# convert to BGR only if preview enabled or mediapipe needs RGB after scaling
			ok = True
		except Exception:
			ok = False
	else:
		ok,frame=cap.read();
	if not ok: break

	now = time.time()
	process_this = (now - last_process_time) >= FRAME_INTERVAL
	li=ri=0
	if process_this:
		# Prepare frame for mediapipe (resize smaller if extra scaling requested)
		proc = frame
		if PROCESS_SCALE != 1.0:
			new_w = max(64, int(proc.shape[1]*PROCESS_SCALE))
			new_h = max(48, int(proc.shape[0]*PROCESS_SCALE))
			proc = cv2.resize(proc, (new_w, new_h), interpolation=cv2.INTER_AREA)
		res=mp_face.process(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB))
		if res.multi_face_landmarks:
			last_landmarks = res.multi_face_landmarks[0].landmark
		last_process_time = now

	if last_landmarks is not None:
		lms = last_landmarks
		lr=eye_ratio(lms,L); rr=eye_ratio(lms,R)
		if 'plr' in globals():
			lr=plr*(1-SMOOTH_ALPHA)+lr*SMOOTH_ALPHA; rr=prr*(1-SMOOTH_ALPHA)+rr*SMOOTH_ALPHA
		plr,prr=lr,rr
		li,ri=idx(lr),idx(rr)
	if (li,ri)!=last:
		canvas=Image.new("RGBA",(128,32))
		left=eye_frames[li]; right=eye_frames[ri].transpose(Image.FLIP_LEFT_RIGHT)
		canvas.paste(left,(0,0),left); canvas.paste(right,(64,0),right)
		out = apply_column_filter(canvas)
		matrix.SetImage(out); last=(li,ri)
	if not DISABLE_PREVIEW:
		# Ensure BGR for preview text overlay
		if frame.shape[2] == 3:  # already BGR for OpenCV source
			cv2.putText(frame,f"L{li} R{ri}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
			if 'plr' in globals():
				cv2.putText(frame,f"{plr:.2f}/{prr:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			cv2.imshow('Eyes',frame)
			if cv2.waitKey(1)&0xFF==ord('q'): break
if picam is not None:
	picam.stop()
elif cap is not None:
	cap.release()
cv2.destroyAllWindows()

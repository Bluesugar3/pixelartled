"""Independent eye blink animation on 128x32 matrix using a USB/Webcam only.
Optimized for Raspberry Pi Zero / low-power boards.

Dependencies: pip install pillow opencv-python mediapipe rgbmatrix
"""
from PIL import Image
import os, math, time, gc
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import cv2, mediapipe as mp

# (Removed Picamera2 path for simplicity; webcam-only build)

files=["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
eye_frames=[Image.open(f).convert("RGBA") for f in files]  # single-eye (64x32) art (open->blink)
eye_frames_flipped=[im.transpose(Image.FLIP_LEFT_RIGHT) for im in eye_frames]

# ---------------- Performance Tunables (override via env) ----------------
CAM_WIDTH  = int(os.environ.get("CAM_W", 320))   # lower = faster
CAM_HEIGHT = int(os.environ.get("CAM_H", 240))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL", 0.15))  # seconds between face mesh runs (~6-7 FPS)
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.3))  # lower = smoother
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW", "0") == "1"  # set NO_PREVIEW=1 to hide OpenCV window
REFINE_LANDMARKS = os.environ.get("REFINE", "0") == "1"  # off by default for speed
PROCESS_SCALE = float(os.environ.get("PROC_SCALE", 1.0))  # extra scale factor (e.g. 0.75) on resized frame
LED_MIN_INTERVAL = float(os.environ.get("LED_INTERVAL", 0.07))  # min seconds between LED pushes (~14 FPS)
CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
DISABLE_GC = os.environ.get("DISABLE_GC", "1") == "1"
RESIZE_INTERP = cv2.INTER_AREA if os.environ.get("RESIZE_NEAREST", "0") != "1" else cv2.INTER_NEAREST
# Adaptive blink tuning (enable to auto-calibrate open eye baseline and use relative thresholds)
DYNAMIC_THRESH = os.environ.get("DYNAMIC_THRESH", "1") == "1"
BASELINE_ALPHA = float(os.environ.get("BASELINE_ALPHA", 0.05))  # smoothing for open baseline updates
OPEN_FRAC = float(os.environ.get("OPEN_FRAC", 0.80))     # >= this fraction of baseline => fully open (frame 0)
MID_FRAC  = float(os.environ.get("MID_FRAC", 0.65))      # >= this => mid (frame 1)
LOW_FRAC  = float(os.environ.get("LOW_FRAC", 0.52))      # >= this => almost closed (frame 2)
STABLE_FRAMES = int(os.environ.get("STABLE_FRAMES", 2))  # require this many consecutive classifications before committing LED change

# -------------------------------------------------------------------------

opts = RGBMatrixOptions()
opts.rows = 32
opts.cols = 128
opts.hardware_mapping = "adafruit-hat"
# Replace the failing line:
if hasattr(opts, "disable_hardware_pulsing"):
    opts.disable_hardware_pulsing = True  # reduce CPU jitter
elif hasattr(opts, "no_hardware_pulse"):
    opts.no_hardware_pulse = True  # legacy name
if hasattr(opts, 'brightness'):
	try:
		opts.brightness = max(1,min(100,int(os.environ.get('LED_BRIGHTNESS','70'))))
	except Exception:
		pass
matrix = RGBMatrix(options=opts)
try:
	offscreen = matrix.CreateFrameCanvas()  # double buffer if supported
except Exception:
	offscreen = None

# (Disabled) Damaged columns mitigation. Set COL_FIX to enable: 'black'|'drop_green'|'reduce_green'.
# Leaving BAD_COLUMNS empty to avoid any color distortion by default.
BAD_COLUMNS: tuple[int, ...] = ()
COL_FIX_MODE = os.environ.get('COL_FIX', 'off')

def apply_column_filter(img: Image.Image) -> Image.Image:
	if COL_FIX_MODE == 'off' or not BAD_COLUMNS:
		# Return original (just ensure RGB mode)
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

def idx(r, prev=None):
	"""Return eye frame index with hysteresis to avoid rapid flicker.
	Base thresholds (open->closed): 0:>0.30,1:>0.24,2:>0.19, else 3.
	We shift thresholds depending on prior state to stabilize transitions.
	"""
	if prev is None:
		return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3
	if prev == 0:
		return 0 if r>0.28 else 1 if r>0.23 else 2 if r>0.18 else 3
	if prev == 1:
		return 0 if r>0.31 else 1 if r>0.22 else 2 if r>0.18 else 3
	if prev == 2:
		return 0 if r>0.31 else 1 if r>0.25 else 2 if r>0.17 else 3
	# prev == 3 (closed)
	return 0 if r>0.31 else 1 if r>0.25 else 2 if r>0.20 else 3

def classify_dynamic(r: float, baseline: float, prev: int|None) -> int:
	"""Adaptive classification using relative ratio vs. calibrated open baseline.
	Applies light hysteresis by nudging thresholds depending on prior state.
	"""
	if baseline <= 1e-6:  # fallback if not yet calibrated
		return idx(r, prev)
	rel = r / baseline
	# hysteresis adjustments
	adj = 0.0
	if prev == 0:
		adj = -0.02  # make it a bit easier to leave fully open
	elif prev == 3:
		adj = 0.02   # require a bit more to pop open from closed
	of = max(0.0, min(1.2, OPEN_FRAC + adj))
	mf = max(0.0, min(of - 0.02, MID_FRAC + adj))
	lf = max(0.0, min(mf - 0.02, LOW_FRAC + adj))
	if rel >= of:
		return 0
	if rel >= mf:
		return 1
	if rel >= lf:
		return 2
	return 3

# Initialize USB/Webcam
cap = cv2.VideoCapture(CAM_INDEX)
if cap.isOpened():
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
	cap.set(cv2.CAP_PROP_FPS, 15)
	try:  # reduce internal buffering latency if backend supports it
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	except Exception:
		pass
else:
	raise RuntimeError("Could not open camera index %d" % CAM_INDEX)

# Single-threaded OpenCV (reduces overhead on single-core Pi Zero)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

last_process_time = 0.0
last_landmarks = None
last=(-1,-1)
last_led_time = 0.0
prev_li = None
prev_ri = None

# Adaptive baseline & stability tracking
baseline_l = 0.0
baseline_r = 0.0
committed_li = -1
committed_ri = -1
li_stable = 0
ri_stable = 0

if DISABLE_GC:
	try:
		gc.disable()
	except Exception:
		pass
while True:
	# Grab a frame
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
			proc = cv2.resize(proc, (new_w, new_h), interpolation=RESIZE_INTERP)
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
		if DYNAMIC_THRESH:
			# update baselines when eyes appear open (high ratios) OR baseline uninitialized
			if lr > baseline_l or baseline_l == 0.0:
				baseline_l = lr if baseline_l == 0.0 else (baseline_l*(1-BASELINE_ALPHA) + lr*BASELINE_ALPHA)
			if rr > baseline_r or baseline_r == 0.0:
				baseline_r = rr if baseline_r == 0.0 else (baseline_r*(1-BASELINE_ALPHA) + rr*BASELINE_ALPHA)
			li = classify_dynamic(lr, baseline_l, prev_li)
			ri = classify_dynamic(rr, baseline_r, prev_ri)
		else:
			li = idx(lr, prev_li)
			ri = idx(rr, prev_ri)
	else:
		li=ri=0

	# Stability filtering: require consecutive frames before committing a new state
	if li != prev_li:
		li_stable = 1
	else:
		li_stable += 1
	if ri != prev_ri:
		ri_stable = 1
	else:
		ri_stable += 1

	# Commit if stable enough
	if li_stable >= STABLE_FRAMES:
		committed_li = li
	if ri_stable >= STABLE_FRAMES:
		committed_ri = ri

	# Rate-limit LED updates and only push on change of committed states
	if (committed_li, committed_ri) != last and (now - last_led_time) >= LED_MIN_INTERVAL and committed_li >=0 and committed_ri >=0:
		canvas=Image.new("RGBA",(128,32))
		left=eye_frames[committed_li]; right=eye_frames_flipped[committed_ri]
		canvas.paste(left,(0,0),left); canvas.paste(right,(64,0),right)
		out = apply_column_filter(canvas)
		if offscreen is not None:
			try:
				offscreen.SetImage(out.convert("RGB"))
				offscreen = matrix.SwapOnVSync(offscreen)
			except Exception:
				matrix.SetImage(out.convert("RGB"))
		else:
			matrix.SetImage(out.convert("RGB"))
		last=(committed_li,committed_ri)
		last_led_time = now

	prev_li, prev_ri = li, ri
	if not DISABLE_PREVIEW:
		# Ensure BGR for preview text overlay
		if frame.shape[2] == 3:  # already BGR for OpenCV source
			label = f"L{committed_li if committed_li>=0 else li} R{committed_ri if committed_ri>=0 else ri}"
			cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
			if 'plr' in globals():
				if DYNAMIC_THRESH and baseline_l>0 and baseline_r>0:
					cv2.putText(frame,f"{plr:.2f}/{prr:.2f} bl:{baseline_l:.2f} br:{baseline_r:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
				else:
					cv2.putText(frame,f"{plr:.2f}/{prr:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			cv2.imshow('Eyes',frame)
			if cv2.waitKey(1)&0xFF==ord('q'): break
if cap is not None:
	cap.release()
cv2.destroyAllWindows()

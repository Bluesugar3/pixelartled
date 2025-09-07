"""Eye blink display on 128x32 RGB matrix using a single USB camera.

Environment variables (optional):
	CAM_W, CAM_H          Capture resolution (default 160x120 lower = faster)
	PROCESS_INTERVAL      Seconds between landmark runs (default 0.10 faster)
	SMOOTH_ALPHA          Exponential smoothing factor (default 0.3)
	NO_PREVIEW=1          Disable OpenCV preview window
	REFINE=1              Enable iris/extra landmarks (slower)
	LED_INTERVAL          Base seconds between LED pushes (default 0.07)
	CAM_INDEX             Camera index (default 0)
	LED_BRIGHTNESS        1..100 (default 70)
	LED_SLOWDOWN          Multiplier to slow steady LED updates (default 1.5)
	BLINK_LED_FACTOR      Multiplier (<1) to speed LED push during active blink (default 0.4)
	BLINK_DELTA           Minimum ratio change to treat as fast blink (default 0.010)
	BLINK_ACCEL=1         Enable frame skipping acceleration during rapid close/open
	PROC_SCALE            Extra downscale for inference (default 1.0; e.g. 0.7)
	PROC_W / PROC_H       Explicit processing size (overrides PROC_SCALE if both >0)
	CAM_FPS               Attempted webcam FPS (default 15; try 10 or 8 on weak boards)
"""

from PIL import Image
import os, math, time
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import cv2, mediapipe as mp

# ------------------------------------------------------------
# Load eye animation frames (indexes: 0 open -> 3 closed)
# ------------------------------------------------------------
EYE_FILES = ["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
EYES_L = [Image.open(f).convert("RGBA") for f in EYE_FILES]
EYES_R = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in EYES_L]

# ------------------------------------------------------------
# Config (env overrides)
# ------------------------------------------------------------
CAM_WIDTH  = int(os.environ.get("CAM_W", 160))
CAM_HEIGHT = int(os.environ.get("CAM_H", 120))
CAM_FPS = int(os.environ.get("CAM_FPS", 15))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL", 0.10))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.3))
LED_MIN_INTERVAL = float(os.environ.get("LED_INTERVAL", 0.07))
LED_SLOWDOWN = float(os.environ.get("LED_SLOWDOWN", 1.5))
BLINK_LED_FACTOR = float(os.environ.get("BLINK_LED_FACTOR", 0.4))
BLINK_DELTA = float(os.environ.get("BLINK_DELTA", 0.010))
BLINK_ACCEL = os.environ.get("BLINK_ACCEL", "1") == "1"
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW", "0") == "1"
REFINE_LANDMARKS = os.environ.get("REFINE", "0") == "1"
CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
BRIGHTNESS = max(1, min(100, int(os.environ.get("LED_BRIGHTNESS", "70"))))
# Dynamic calibration window & tolerance (percentage of range to extend)
CAL_WINDOW = int(os.environ.get("CAL_WINDOW", 60))       # number of recent frames to track per eye
CAL_TOL = float(os.environ.get("CAL_TOL", 0.05))          # 0.05 = extend 5% on both ends of range
CAL_MIN_RANGE = float(os.environ.get("CAL_MIN_RANGE", 0.03))  # fallback to static thresholds until range >= this
PROC_SCALE = float(os.environ.get("PROC_SCALE", 1.0))
PROC_W = int(os.environ.get("PROC_W", 0))
PROC_H = int(os.environ.get("PROC_H", 0))
T_OPEN = float(os.environ.get("T_OPEN", 0.45))  # ratio above this = fully open frame 0
T_HALF = float(os.environ.get("T_HALF", 0.35))  # between T_HALF..T_OPEN = frame 1
T_LOW  = float(os.environ.get("T_LOW", 0.25))   # between T_LOW..T_HALF = frame 2, else frame 3
T_OPEN_L = float(os.environ.get("T_OPEN_L", T_OPEN))
T_HALF_L = float(os.environ.get("T_HALF_L", T_HALF))
T_LOW_L  = float(os.environ.get("T_LOW_L",  T_LOW))
T_OPEN_R = float(os.environ.get("T_OPEN_R", T_OPEN))
T_HALF_R = float(os.environ.get("T_HALF_R", T_HALF))
T_LOW_R  = float(os.environ.get("T_LOW_R",  T_LOW))
SHOW_RAW = os.environ.get("SHOW_RAW", "0") == "1"

# ------------------------------------------------------------
# FaceMesh + landmark indices (left/right from viewer perspective)
# Landmark order comment: [outer, top1, top2, inner, bottom2, bottom1]
# ------------------------------------------------------------
L_IDS = [362,385,387,263,373,380]   # left eye
R_IDS = [33,160,158,133,153,144]    # right eye

def eye_ratio(lms, ids):
	pts = [lms[i] for i in ids]
	dist = lambda a,b: math.dist((a.x,a.y),(b.x,b.y))
	vert = (dist(pts[1],pts[5]) + dist(pts[2],pts[4])) / 2.0
	horiz = dist(pts[0],pts[3])
	return vert / horiz if horiz else 0.0

def classify_static(r: float, left: bool) -> int:
	"""Static classification; allows per-eye overrides."""
	if left:
		return 0 if r > T_OPEN_L else 1 if r > T_HALF_L else 2 if r > T_LOW_L else 3
	return 0 if r > T_OPEN_R else 1 if r > T_HALF_R else 2 if r > T_LOW_R else 3

def adaptive_index(ratio, history, prev_idx, prev_ratio=None):
	"""Adaptive classification using dynamic min/max from recent history.

	ratio: current (smoothed) openness metric.
	history: list of recent ratios (most recent last) – mutated in-place.
	prev_idx: previous frame index (for mild hysteresis if needed later).

	Returns frame index 0..3.
	"""
	# Maintain rolling window
	history.append(ratio)
	if len(history) > CAL_WINDOW:
		history.pop(0)

	r_min = min(history)
	r_max = max(history)
	rng = r_max - r_min
	if rng < CAL_MIN_RANGE or len(history) < 5:
		# Not enough spread yet -> use static thresholds
		return classify_static(ratio, True)

	# Expand range slightly for tolerance
	expand = rng * CAL_TOL
	adj_min = r_min - expand
	adj_max = r_max + expand
	if adj_max - adj_min <= 0:
		return classify_static(ratio, True)
	# Clamp & normalize 0..1
	r_clamped = max(adj_min, min(adj_max, ratio))
	n = (r_clamped - adj_min) / (adj_max - adj_min)

	# Map normalized openness to frame index baseline (open->closed)
	if n >= 0.75: base = 0
	elif n >= 0.50: base = 1
	elif n >= 0.25: base = 2
	else: base = 3

	# Blink acceleration: if rapid change detected, move one extra frame toward direction
	if prev_ratio is not None and prev_idx is not None:
		delta = prev_ratio - ratio  # positive if closing
		if BLINK_ACCEL:
			if delta > BLINK_DELTA and base > prev_idx:
				base = min(3, base + 1)
			elif delta < -BLINK_DELTA and base < prev_idx:
				base = max(0, base - 1)
	return base

def init_matrix():
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
	matrix = RGBMatrix(options=opts)
	try:
		offscreen = matrix.CreateFrameCanvas()
	except Exception:
		offscreen = None
	return matrix, offscreen

def compose(left_idx, right_idx):
	canvas = Image.new("RGBA", (128,32))
	L = EYES_L[left_idx]
	R = EYES_R[right_idx]
	canvas.paste(L, (0,0), L)
	canvas.paste(R, (64,0), R)
	return canvas.convert("RGB")

def open_camera():
	cap = cv2.VideoCapture(CAM_INDEX)
	if not cap.isOpened():
		raise RuntimeError(f"Unable to open camera index {CAM_INDEX}")
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
	cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
	try:
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	except Exception:
		pass
	try:
		cv2.setNumThreads(1)
	except Exception:
		pass
	return cap

def main():
	matrix, offscreen = init_matrix()
	cap = open_camera()
	face_mesh = mp.solutions.face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=REFINE_LANDMARKS,
		static_image_mode=False,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
	)

	last_lms = None
	last_proc_t = 0.0
	last_led_t = 0.0
	last_pair = (-1,-1)
	smooth_l = smooth_r = None
	prev_li = prev_ri = None
	prev_ratio_l = prev_ratio_r = None
	hist_l: list[float] = []
	hist_r: list[float] = []
	base_interval = LED_MIN_INTERVAL * LED_SLOWDOWN

	while True:
		ok, frame = cap.read()
		if not ok:
			break
		now = time.time()

		if (now - last_proc_t) >= FRAME_INTERVAL:
			proc = frame
			# Apply explicit processing resolution first, else scaling
			if PROC_W > 0 and PROC_H > 0:
				proc = cv2.resize(proc,(PROC_W,PROC_H),interpolation=cv2.INTER_AREA)
			elif PROC_SCALE != 1.0:
				new_w = max(32,int(proc.shape[1]*PROC_SCALE))
				new_h = max(24,int(proc.shape[0]*PROC_SCALE))
				proc = cv2.resize(proc,(new_w,new_h),interpolation=cv2.INTER_AREA)
			rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
			res = face_mesh.process(rgb)
			if res.multi_face_landmarks:
				last_lms = res.multi_face_landmarks[0].landmark
			last_proc_t = now

		li = ri = 0
		if last_lms is not None:
			lr = eye_ratio(last_lms, L_IDS)
			rr = eye_ratio(last_lms, R_IDS)
			if smooth_l is None:
				smooth_l, smooth_r = lr, rr
			else:
				smooth_l = smooth_l*(1-SMOOTH_ALPHA) + lr*SMOOTH_ALPHA
				smooth_r = smooth_r*(1-SMOOTH_ALPHA) + rr*SMOOTH_ALPHA
			li = adaptive_index(smooth_l, hist_l, prev_li, prev_ratio_l)
			ri = adaptive_index(smooth_r, hist_r, prev_ri, prev_ratio_r)
		prev_li, prev_ri = li, ri
		prev_ratio_l, prev_ratio_r = smooth_l, smooth_r

		# Blink-active = change in index; push faster with BLINK_LED_FACTOR
		blink_active = (li,ri) != last_pair and (prev_li is not None and (li != prev_li or ri != prev_ri))
		interval = base_interval if not blink_active else min(base_interval, LED_MIN_INTERVAL * BLINK_LED_FACTOR)
		if (li,ri) != last_pair and (now - last_led_t) >= interval:
			img = compose(li, ri)
			if offscreen is not None:
				try:
					offscreen.SetImage(img)
					offscreen = matrix.SwapOnVSync(offscreen)
				except Exception:
					matrix.SetImage(img)
			else:
				matrix.SetImage(img)
			last_pair = (li,ri)
			last_led_t = now

		if not DISABLE_PREVIEW:
			cv2.putText(frame, f"L{li} R{ri}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
			if smooth_l is not None and SHOW_RAW:
				cv2.putText(frame, f"{smooth_l:.3f}/{smooth_r:.3f}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
			cv2.imshow("Eyes", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

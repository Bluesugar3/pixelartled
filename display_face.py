"""Eye blink display on 128x32 RGB matrix using a single USB camera.

Environment variables (optional):
  CAM_W, CAM_H          Capture resolution (default 320x240)
  PROCESS_INTERVAL      Seconds between landmark runs (default 0.15)
  SMOOTH_ALPHA          Exponential smoothing factor (default 0.3)
  NO_PREVIEW=1          Disable OpenCV preview window
  REFINE=1              Enable iris/extra landmarks (slower)
  LED_INTERVAL          Minimum seconds between LED updates (default 0.07)
  CAM_INDEX             Camera index (default 0)
  LED_BRIGHTNESS        1..100 (default 70)
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
CAM_WIDTH  = int(os.environ.get("CAM_W", 320))
CAM_HEIGHT = int(os.environ.get("CAM_H", 240))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL", 0.15))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.3))
LED_MIN_INTERVAL = float(os.environ.get("LED_INTERVAL", 0.07))
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW", "0") == "1"
REFINE_LANDMARKS = os.environ.get("REFINE", "0") == "1"
CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
BRIGHTNESS = max(1, min(100, int(os.environ.get("LED_BRIGHTNESS", "70"))))
ADAPTIVE = os.environ.get("ADAPTIVE", "0") == "1"  # dynamic per-eye range
STEP_TRANSITIONS = os.environ.get("STEP_TRANSITIONS", "1") == "1"  # slow eye frame changes

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

def classify_static(r, prev):
	"""Static threshold classification with mild hysteresis (legacy mode)."""
	if prev is None:
		return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3
	if prev == 0:
		return 0 if r>0.28 else 1 if r>0.23 else 2 if r>0.18 else 3
	if prev == 3:
		return 0 if r>0.31 else 1 if r>0.25 else 2 if r>0.20 else 3
	return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

class AdaptiveStats:
	__slots__ = ("min","max","warm")
	def __init__(self):
		self.min =  1e9
		self.max = -1e9
		self.warm = 0  # number of updates

def update_stats(stats: AdaptiveStats, value: float):
	# Exponential style adaptation: only expand envelope (slow contraction)
	if value < stats.min:
		stats.min = value
	else:
		# allow slow drift downward
		stats.min = stats.min*0.999 + min(value, stats.min)*0.001
	if value > stats.max:
		stats.max = value
	else:
		stats.max = stats.max*0.999 + max(value, stats.max)*0.001
	stats.warm += 1

def classify_adaptive(r: float, prev: int | None, stats: AdaptiveStats):
	"""Adaptive classification: map ratio into 0..1 using running min/max.
	Then segment into 4 frames with hysteresis margins.
	"""
	rng = stats.max - stats.min
	if stats.warm < 15 or rng < 0.02:  # not calibrated; fallback static
		return classify_static(r, prev)
	norm = (r - stats.min) / (rng + 1e-6)
	# Base cut points in normalized space
	cuts = [0.75, 0.50, 0.25]  # >0.75 fully open, etc.
	# Hysteresis: shift cuts a bit depending on previous frame
	if prev == 0:  # make it harder to leave open
		cuts = [c - 0.03 for c in cuts]
	elif prev == 3:  # make it harder to leave closed
		cuts = [c + 0.03 for c in cuts]
	if norm > cuts[0]:
		return 0
	if norm > cuts[1]:
		return 1
	if norm > cuts[2]:
		return 2
	return 3

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
	cap.set(cv2.CAP_PROP_FPS, 15)
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
	smooth_l = None
	smooth_r = None
	prev_li = prev_ri = None
	# For adaptive mode
	l_stats = AdaptiveStats() if ADAPTIVE else None
	r_stats = AdaptiveStats() if ADAPTIVE else None
	# Display (stepped) frames vs target frames
	disp_li = disp_ri = 0

	while True:
		ok, frame = cap.read()
		if not ok:
			break
		now = time.time()

		if (now - last_proc_t) >= FRAME_INTERVAL:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
			if ADAPTIVE:
				update_stats(l_stats, smooth_l)
				update_stats(r_stats, smooth_r)
				li = classify_adaptive(smooth_l, prev_li, l_stats)
				ri = classify_adaptive(smooth_r, prev_ri, r_stats)
			else:
				li = classify_static(smooth_l, prev_li)
				ri = classify_static(smooth_r, prev_ri)
		prev_li, prev_ri = li, ri

		# Step transitions: move only one frame toward target per update
		if STEP_TRANSITIONS:
			if li > disp_li: disp_li += 1
			elif li < disp_li: disp_li -= 1
			if ri > disp_ri: disp_ri += 1
			elif ri < disp_ri: disp_ri -= 1
		else:
			disp_li, disp_ri = li, ri

		if (disp_li,disp_ri) != last_pair and (now - last_led_t) >= LED_MIN_INTERVAL:
			img = compose(disp_li, disp_ri)
			if offscreen is not None:
				try:
					offscreen.SetImage(img)
					offscreen = matrix.SwapOnVSync(offscreen)
				except Exception:
					matrix.SetImage(img)
			else:
				matrix.SetImage(img)
			last_pair = (disp_li,disp_ri)
			last_led_t = now

		if not DISABLE_PREVIEW:
			cv2.putText(frame, f"L{disp_li}->{li} R{disp_ri}->{ri}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
			if smooth_l is not None:
				if ADAPTIVE and l_stats.warm >= 15:
					cv2.putText(frame, f"{smooth_l:.2f}/{smooth_r:.2f} A", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
				else:
					cv2.putText(frame, f"{smooth_l:.2f}/{smooth_r:.2f}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
			cv2.imshow("Eyes", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

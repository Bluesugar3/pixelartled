
"""Preview-based eye tracking + LED matrix output (optimized for Pi Zero 2 W).

Simplified logic taken from preview_face:
  - Independent eye ratios -> 4-frame eyelid animation (open->closed)
  - Optional smoothing
  - Throttled FaceMesh calls
  - Throttled LED matrix updates

Environment (optional):
  CAM_W,CAM_H        Capture size (default 160x120)
  CAM_INDEX          Camera index (default 0)
  PROCESS_INTERVAL   Seconds between landmark runs (default 0.12)
  LED_INTERVAL       Min seconds between LED pushes (default 0.07)
  SMOOTH_ALPHA       0..1 (default 0.3)
  PROC_SCALE         Extra scale factor for processing (e.g. 0.75) default 1.0
  NO_PREVIEW=1       Disable OpenCV preview window
  REFINE=1           Enable iris refinement (slower)
  LED_BRIGHTNESS     1..100 (default 70)
"""

from PIL import Image
import os, math, time
import cv2, mediapipe as mp
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# ---------------- Assets ----------------
EYE_FILES = ["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
EYES_L = [Image.open(f).convert("RGBA") for f in EYE_FILES]
EYES_R = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in EYES_L]

# ---------------- Config ----------------
CAM_W = int(os.environ.get("CAM_W","160"))
CAM_H = int(os.environ.get("CAM_H","120"))
CAM_INDEX = int(os.environ.get("CAM_INDEX","0"))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL","0.12"))
LED_INTERVAL = float(os.environ.get("LED_INTERVAL","0.07"))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA","0.3"))
PROC_SCALE = float(os.environ.get("PROC_SCALE","1.0"))
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW","0") == "1"
REFINE = os.environ.get("REFINE","0") == "1"
BRIGHTNESS = max(1,min(100,int(os.environ.get("LED_BRIGHTNESS","70"))))

# Landmark indices (camera -> subject corrected)
L_IDS = [362,385,387,263,373,380]
R_IDS = [33,160,158,133,153,144]

def eye_ratio(lms, ids):
	pts=[lms[i] for i in ids]
	d=lambda a,b: math.dist((a.x,a.y),(b.x,b.y))
	v=(d(pts[1],pts[5])+d(pts[2],pts[4]))/2
	h=d(pts[0],pts[3])
	return v/h if h else 0

def classify(r):
	# Same static thresholds as preview_face
	return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

def init_matrix():
	opts=RGBMatrixOptions()
	opts.rows=32; opts.cols=128; opts.hardware_mapping="adafruit-hat"
	if hasattr(opts,"disable_hardware_pulsing"): opts.disable_hardware_pulsing=True
	elif hasattr(opts,"no_hardware_pulse"): opts.no_hardware_pulse=True
	if hasattr(opts,"brightness"): opts.brightness=BRIGHTNESS
	matrix=RGBMatrix(options=opts)
	try: off=matrix.CreateFrameCanvas()
	except Exception: off=None
	return matrix,off

def compose(li,ri):
	canvas=Image.new("RGBA",(128,32))
	L=EYES_L[li]; R=EYES_R[ri]
	canvas.paste(L,(0,0),L)
	canvas.paste(R,(64,0),R)
	return canvas.convert("RGB")

def open_camera():
	cap=cv2.VideoCapture(CAM_INDEX)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open camera {CAM_INDEX}")
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
	cap.set(cv2.CAP_PROP_FPS,15)
	try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
	except Exception: pass
	try: cv2.setNumThreads(1)
	except Exception: pass
	return cap

def main():
	matrix,off=init_matrix()
	cap=open_camera()
	face=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=REFINE)

	last_proc=0.0
	last_led=0.0
	last=(-1,-1)
	sm_l=sm_r=None,None

	while True:
		ok,frame=cap.read()
		if not ok: break
		now=time.time()

		res=None
		if (now - last_proc) >= FRAME_INTERVAL or last_proc==0.0:
			proc=frame
			if PROC_SCALE!=1.0:
				new_w=max(64,int(proc.shape[1]*PROC_SCALE))
				new_h=max(48,int(proc.shape[0]*PROC_SCALE))
				proc=cv2.resize(proc,(new_w,new_h),interpolation=cv2.INTER_AREA)
			res=face.process(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB))
			last_proc=now

		li=ri=0
		if res and res.multi_face_landmarks:
			lms=res.multi_face_landmarks[0].landmark
			lr=eye_ratio(lms,L_IDS); rr=eye_ratio(lms,R_IDS)
			pl,pr=sm_l
			if pl is None:
				sm_l=(lr,rr)
			else:
				lr = pl*(1-SMOOTH_ALPHA) + lr*SMOOTH_ALPHA
				rr = pr*(1-SMOOTH_ALPHA) + rr*SMOOTH_ALPHA
				sm_l=(lr,rr)
			li,ri=classify(lr),classify(rr)

		if (li,ri)!=last and (now - last_led) >= LED_INTERVAL:
			img=compose(li,ri)
			if off is not None:
				try:
					off.SetImage(img)
					off=matrix.SwapOnVSync(off)
				except Exception:
					matrix.SetImage(img)
			else:
				matrix.SetImage(img)
			last=(li,ri)
			last_led=now

		if not DISABLE_PREVIEW:
			cv2.putText(frame,f"L{li} R{ri}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
			if sm_l[0] is not None:
				cv2.putText(frame,f"{sm_l[0]:.2f}/{sm_l[1]:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			cv2.imshow('Eyes',frame)
			if cv2.waitKey(1)&0xFF==ord('q'): break

	cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
	main()

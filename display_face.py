
"""Preview-based eye tracking + LED matrix output (optimized for Pi Zero 2 W).

Merged advanced camera / processing optimizations from preview_face:
	- Independent eye ratios -> 4 eyelid frames
	- Smoothing + unilateral blink stabilization
	- Throttled landmark processing (PROCESS_INTERVAL)
	- Throttled LED updates (LED_INTERVAL)
	- Optional preprocessing (CLAHE, bilateral denoise)
	- Optional processing downscale (PROC_SCALE)
	- Threaded capture to reduce latency (CAPTURE_THREAD)
	- Adaptive refine toggle (ADAPT_REFINE) auto-disables iris refinement if too slow
	- MJPG FourCC request for higher USB cam FPS (FOURCC_MJPG)
	- FPS / status overlay when preview enabled (SHOW_FPS)

Environment (optional):
	CAM_W,CAM_H          Capture size (default 160x120)
	CAM_FPS              Requested camera FPS hint (default 15)
	CAM_INDEX            Camera index (default 0)
	PROCESS_INTERVAL     Seconds between landmark runs (default 0.12; 0 = every frame)
	LED_INTERVAL         Min seconds between LED pushes (default 0.07)
	SMOOTH_ALPHA         0..1 smoothing (default 0.3)
	PROC_SCALE           Processing scale factor (0.2..1.0) (default 1.0)
	NO_PREVIEW=1         Disable OpenCV preview window
	REFINE=1             Enable iris refinement at startup (slower)
	ADAPT_REFINE=1       Allow auto disable of refine if average proc time too high
	TARGET_PROC_FPS      Target landmark FPS for adaptive refine (default 15)
	CAPTURE_THREAD=1     Enable threaded capture (OpenCV path)
	CLAHE=1              Apply local contrast enhancement
	DENOISE=1            Mild bilateral filter before FaceMesh
	SHOW_FPS=1           Show FPS / status overlay
	FOURCC_MJPG=1        Try setting MJPG FourCC on camera
	LED_BRIGHTNESS       1..100 (default 100)
	STABILIZE=1          Enable unilateral blink stabilization
	STABLE_DELTA         Threshold for change detection (default 0.015)
	ASYM_RATIO           Asymmetry ratio for unilateral blink logic (default 0.60)
"""

from PIL import Image
import os, math, time, threading, queue
import cv2, mediapipe as mp, numpy as np
from rgbmatrix import RGBMatrix, RGBMatrixOptions

# ---------------- Assets ----------------
EYE_FILES = ["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
EYES_L = [Image.open(f).convert("RGBA") for f in EYE_FILES]
EYES_R = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in EYES_L]

# ---------------- Config ----------------
CAM_W = int(os.environ.get("CAM_W","160"))
CAM_H = int(os.environ.get("CAM_H","120"))
CAM_INDEX = int(os.environ.get("CAM_INDEX","0"))
CAM_FPS = int(os.environ.get("CAM_FPS","15"))
FRAME_INTERVAL = float(os.environ.get("PROCESS_INTERVAL","0.12"))
LED_INTERVAL = float(os.environ.get("LED_INTERVAL","0.07"))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA","0.3"))
PROC_SCALE = float(os.environ.get("PROC_SCALE","1.0"))
PROC_SCALE = max(0.2,min(1.0,PROC_SCALE))
DISABLE_PREVIEW = os.environ.get("NO_PREVIEW","0") == "1"
REFINE = os.environ.get("REFINE","0") == "1"
ADAPT_REFINE = os.environ.get("ADAPT_REFINE","1") == "1" and REFINE
TARGET_PROC_FPS = float(os.environ.get("TARGET_PROC_FPS","15"))
CAPTURE_THREAD = os.environ.get("CAPTURE_THREAD","1") == "1"
CLAHE = os.environ.get("CLAHE","0") == "1"
DENOISE = os.environ.get("DENOISE","0") == "1"
SHOW_FPS = os.environ.get("SHOW_FPS","0") == "1"
FOURCC_MJPG = os.environ.get("FOURCC_MJPG","1") == "1"
BRIGHTNESS = max(1,min(100,int(os.environ.get("LED_BRIGHTNESS","100"))))
STABILIZE = os.environ.get("STABILIZE","1") == "1"
STABLE_DELTA = float(os.environ.get("STABLE_DELTA","0.015"))
ASYM_RATIO = float(os.environ.get("ASYM_RATIO","0.60"))

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
	cap.set(cv2.CAP_PROP_FPS,CAM_FPS)
	if FOURCC_MJPG:
		try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
		except Exception: pass
	try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
	except Exception: pass
	try: cv2.setNumThreads(1)
	except Exception: pass
	return cap

def build_facemesh(refine: bool):
	return mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=refine)

def main():
	matrix,off=init_matrix()
	cap=open_camera()
	face=build_facemesh(REFINE)

	last_proc=0.0
	last_led=0.0
	last=(-1,-1)
	sm_l=(None,None)

	# Performance tracking
	proc_times=[]
	_fps=0.0; _frames=0; _t0=time.time()

	# Threaded capture
	frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
	stop_flag=False

	def capture_loop():
		while not stop_flag:
			ok,f=cap.read()
			if not ok:
				time.sleep(0.01); continue
			if frame_q.full():
				try: frame_q.get_nowait()
				except Exception: pass
			try: frame_q.put_nowait(f)
			except Exception: pass

	th=None
	if CAPTURE_THREAD:
		th=threading.Thread(target=capture_loop,daemon=True)
		th.start()

	cur_li=cur_ri=0
	adapt_msg=None

	try:
		while True:
			if CAPTURE_THREAD and th is not None:
				try:
					frame=frame_q.get(timeout=1.0)
					ok=True
				except Exception:
					ok=False
			else:
				ok,frame=cap.read()
			if not ok: break
			now=time.time()

			do_proc = (FRAME_INTERVAL==0) or ((now-last_proc)>=FRAME_INTERVAL) or last_proc==0.0
			li=cur_li; ri=cur_ri
			if do_proc:
				proc=frame
				if PROC_SCALE!=1.0:
					new_w=max(64,int(proc.shape[1]*PROC_SCALE))
					new_h=max(48,int(proc.shape[0]*PROC_SCALE))
					proc=cv2.resize(proc,(new_w,new_h),interpolation=cv2.INTER_AREA)
				if CLAHE:
					lab=cv2.cvtColor(proc,cv2.COLOR_BGR2LAB)
					l,a,b=cv2.split(lab)
					clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
					l=clahe.apply(l)
					lab=cv2.merge((l,a,b))
					proc=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
				if DENOISE:
					proc=cv2.bilateralFilter(proc,5,50,50)
				start_t=time.time()
				res=face.process(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB))
				if res and res.multi_face_landmarks:
					lms=res.multi_face_landmarks[0].landmark
					lr=eye_ratio(lms,L_IDS); rr=eye_ratio(lms,R_IDS)
					pl,pr=sm_l
					if pl is None:
						sm_l=(lr,rr)
					else:
						orig_lr,orig_rr=lr,rr
						lr=pl*(1-SMOOTH_ALPHA)+lr*SMOOTH_ALPHA
						rr=pr*(1-SMOOTH_ALPHA)+rr*SMOOTH_ALPHA
						if STABILIZE:
							chg_l=abs(orig_lr-pl); chg_r=abs(orig_rr-pr)
							if chg_r>STABLE_DELTA and chg_l<STABLE_DELTA and (pr-orig_rr)>STABLE_DELTA:
								lr=pl
							elif chg_l>STABLE_DELTA and chg_r<STABLE_DELTA and (pl-orig_lr)>STABLE_DELTA:
								rr=pr
							if pr>0 and (rr/max(lr,1e-6))<ASYM_RATIO and chg_l<STABLE_DELTA:
								lr=pl
							if pl>0 and (lr/max(rr,1e-6))<ASYM_RATIO and chg_r<STABLE_DELTA:
								rr=pr
						sm_l=(lr,rr)
					li,ri=classify(lr),classify(rr)
					cur_li,cur_ri=li,ri
				end_t=time.time(); proc_times.append(end_t-start_t)
				if len(proc_times)>60: proc_times.pop(0)
				last_proc=now
				if ADAPT_REFINE and REFINE and len(proc_times)>=30:
					avg=sum(proc_times)/len(proc_times)
					if avg > (1.0/max(1.0,TARGET_PROC_FPS))*1.25:
						try: face.close()
						except Exception: pass
						REFINE=False
						face=build_facemesh(False)
						adapt_msg="refine->off"; proc_times.clear()

			if (li,ri)!=last and (now-last_led)>=LED_INTERVAL:
				img=compose(li,ri)
				if off is not None:
					try:
						off.SetImage(img)
						off=matrix.SwapOnVSync(off)
					except Exception:
						matrix.SetImage(img)
				else:
					matrix.SetImage(img)
				last=(li,ri); last_led=now

			if not DISABLE_PREVIEW:
				cv2.putText(frame,f"L{li} R{ri}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),1)
				if sm_l[0] is not None:
					cv2.putText(frame,f"{sm_l[0]:.2f}/{sm_l[1]:.2f}",(8,44),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
				_frames+=1
				if SHOW_FPS and (_frames%15)==0:
					now_t=time.time(); dt=now_t-_t0; _fps=15.0/dt if dt>0 else 0; _t0=now_t
				if SHOW_FPS:
					cv2.putText(frame,f"{_fps:4.1f}fps s={PROC_SCALE}",(8,64),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
					if adapt_msg:
						cv2.putText(frame,adapt_msg,(8,82),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,200,255),1)
				cv2.imshow('Eyes',frame)
				if cv2.waitKey(1)&0xFF==ord('q'): break
	finally:
		if CAPTURE_THREAD and th is not None:
			stop_flag=True
		cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
	main()

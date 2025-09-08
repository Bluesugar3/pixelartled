"""Desktop preview with independent left/right eye frames.
Deps: pip install pillow opencv-python mediapipe
On Raspberry Pi (Arducam/RPi Cam v2), also install: sudo apt install python3-picamera2

Quality / performance tuning (optional via environment variables):
	CAM_W, CAM_H          Requested capture size (default 640x480 if supported)
	CAM_FPS               Target FPS hint (default 30)
	PROC_SCALE            0<scale<=1.0; scale frame before FaceMesh (default 1.0). Lower = faster
	CLAHE=1               Apply local contrast enhancement before FaceMesh
	DENOISE=1             Light bilateral denoise before FaceMesh
	SMOOTH_ALPHA          0..1 smoothing strength for eye ratio (default 0.3)
	SHOW_FPS=1            Overlay FPS + processing scale
	REFINE=0/1            mediapipe iris refinement (1 slower, default 1 to match previous behavior)

All enhancements are optional; disabling (unset) yields behavior close to original.
"""
from PIL import Image
import cv2, mediapipe as mp, math, numpy as np, os, time, threading, queue
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
eye_frames=[Image.open(f).convert("RGBA") for f in files]

# ---------------- Config (env) ----------------
CAM_W=int(os.environ.get("CAM_W","640"))
CAM_H=int(os.environ.get("CAM_H","480"))
CAM_FPS=int(os.environ.get("CAM_FPS","30"))
PROC_SCALE=float(os.environ.get("PROC_SCALE","1.0"))
PROC_SCALE=max(0.2,min(1.0,PROC_SCALE))  # clamp reasonable
CLAHE= os.environ.get("CLAHE","0") == "1"
DENOISE= os.environ.get("DENOISE","0") == "1"
SMOOTH_ALPHA=float(os.environ.get("SMOOTH_ALPHA","0.3"))
SHOW_FPS= os.environ.get("SHOW_FPS","0") == "1"
REFINE= os.environ.get("REFINE","1") == "1"  # keep previous default True
PROCESS_INTERVAL = float(os.environ.get("PROCESS_INTERVAL","0"))  # seconds; 0 = every frame
CAPTURE_THREAD = os.environ.get("CAPTURE_THREAD","1") == "1"
TARGET_PROC_FPS = float(os.environ.get("TARGET_PROC_FPS","15"))  # for adaptive refine toggle
ADAPT_REFINE = os.environ.get("ADAPT_REFINE","1") == "1" and REFINE  # allow auto disable
FOURCC_MJPG = os.environ.get("FOURCC_MJPG","1") == "1"  # try MJPG for higher raw fps

def build_facemesh(refine: bool):
	return mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=refine)

mp_face=build_facemesh(REFINE)
# Swap to match display orientation (camera vs subject)
L=[362,385,387,263,373,380]          # left eye (display-left)
R=[33,160,158,133,153,144]           # right eye (display-right)
def eye_ratio(l,i):
	pts=[l[x] for x in i]; d=lambda a,b:math.dist((a.x,a.y),(b.x,b.y))
	v=(d(pts[1],pts[5])+d(pts[2],pts[4]))/2; h=d(pts[0],pts[3]); return v/h if h else 0
def idx(r): return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

picam = None  # type: ignore[assignment]
cap = None  # type: ignore[assignment]
if _HAVE_PICAMERA2:
	try:
		picam = Picamera2()
		RES = (CAM_W, CAM_H)
		cfg = picam.create_video_configuration(main={"size": RES})
		picam.configure(cfg)
		picam.start()
	except Exception:
		picam=None
if picam is None:
	cap=cv2.VideoCapture(0)
	if cap.isOpened():
		# Try to set properties; failures are fine
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
		cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
		if FOURCC_MJPG:
			try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
			except Exception: pass
		try: cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
		except Exception: pass
		try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
		except Exception: pass
		try: cv2.setNumThreads(1)  # reduce CPU jitter
		except Exception: pass
last=(-1,-1); scale=8
preview=np.zeros((32*scale,128*scale,3),dtype=np.uint8)

# FPS tracking
_t0=time.time(); _frames=0; _fps=0.0

# Previous smoothed ratios (floats)
plr=prr=None
last_proc_time=0.0
proc_times=[]  # rolling list for adaptive refine
cur_li=cur_ri=0

# Optional threaded capture (OpenCV path only)
frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
stop_flag=False

def capture_loop():
	while not stop_flag:
		ok, frm = cap.read()
		if not ok:
			time.sleep(0.01)
			continue
		if frame_q.full():
			try: frame_q.get_nowait()
			except Exception: pass
		try: frame_q.put_nowait(frm)
		except Exception:
			pass

if cap is not None and CAPTURE_THREAD and cap.isOpened():
	th=threading.Thread(target=capture_loop,daemon=True)
	th.start()
else:
	th=None

while True:
	if picam is not None:
		try:
			frame = picam.capture_array()
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			ok = True
		except Exception:
			ok = False
	else:
		if CAPTURE_THREAD and th is not None:
			try:
				frame = frame_q.get(timeout=1.0)
				ok = True
			except Exception:
				ok = False
		else:
			ok,frame=cap.read();
	if not ok: break
	proc=frame
	# Downscale for processing speed (FaceMesh uses normalized coords so scaling OK)
	if PROC_SCALE<1.0:
		new_w=max(64,int(proc.shape[1]*PROC_SCALE))
		new_h=max(48,int(proc.shape[0]*PROC_SCALE))
		proc=cv2.resize(proc,(new_w,new_h),interpolation=cv2.INTER_AREA)
	# Optional local contrast enhance (acts mainly on luminance)
	if CLAHE:
		lab=cv2.cvtColor(proc,cv2.COLOR_BGR2LAB)
		l,a,b=cv2.split(lab)
		clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		l=clahe.apply(l)
		lab=cv2.merge((l,a,b))
		proc=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
	# Optional mild denoise
	if DENOISE:
		proc=cv2.bilateralFilter(proc,5,50,50)
	# Throttle FaceMesh if PROCESS_INTERVAL > 0
	do_process = (PROCESS_INTERVAL==0) or (time.time()-last_proc_time >= PROCESS_INTERVAL)
	li=cur_li; ri=cur_ri
	if do_process:
		start_t=time.time()
		res=mp_face.process(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB))
		if res.multi_face_landmarks:
			lms=res.multi_face_landmarks[0].landmark
			lr=eye_ratio(lms,L); rr=eye_ratio(lms,R)
			if plr is not None:
				lr=plr*(1-SMOOTH_ALPHA)+lr*SMOOTH_ALPHA
				rr=prr*(1-SMOOTH_ALPHA)+rr*SMOOTH_ALPHA
			plr,prr=lr,rr
			li,ri=idx(lr),idx(rr); cur_li,cur_ri=li,ri
		end_t=time.time(); proc_times.append(end_t-start_t)
		if len(proc_times)>60: proc_times.pop(0)
		last_proc_time=time.time()
		# Adaptive refine: if too slow, rebuild without refine
		if ADAPT_REFINE and REFINE and len(proc_times)>=30:
			avg=sum(proc_times)/len(proc_times)
			if avg > (1.0/max(1.0,TARGET_PROC_FPS))*1.25:
				try:
					mp_face.close()
				except Exception: pass
				REFINE=False
				mp_face=build_facemesh(False)
				ADAPT_MSG="refine->off"; proc_times.clear()
	if (li,ri)!=last:
		canvas=Image.new("RGBA",(128,32))
		left=eye_frames[li]; right=eye_frames[ri].transpose(Image.FLIP_LEFT_RIGHT)
		canvas.paste(left,(0,0),left); canvas.paste(right,(64,0),right)
		arr=cv2.cvtColor(np.array(canvas.convert("RGB")),cv2.COLOR_RGB2BGR)
		preview=cv2.resize(arr,(128*scale,32*scale),interpolation=cv2.INTER_NEAREST)
		last=(li,ri)
	cv2.putText(preview,f"L{li} R{ri}",(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
	if plr is not None:
		cv2.putText(preview,f"{plr:.2f}/{prr:.2f}",(5,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1)
	# FPS overlay
	_frames+=1
	if SHOW_FPS and (_frames % 15)==0:
		_now=time.time(); dt=_now-_t0; _fps=15.0/dt if dt>0 else 0; _t0=_now
	if SHOW_FPS:
		cv2.putText(preview,f"{_fps:4.1f}fps s={PROC_SCALE}",(5,85),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
		if 'ADAPT_MSG' in globals():
			cv2.putText(preview,ADAPT_MSG,(5,105),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,200,255),1)
	cv2.imshow('Protogen Preview',preview)
	if cv2.waitKey(1)&0xFF==ord('q'): break
if picam is not None:
	picam.stop()
elif cap is not None:
	cap.release()
cv2.destroyAllWindows()
